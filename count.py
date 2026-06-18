#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
猪只计数监控系统 (三线程异步流水线)
基于 YOLO11s (.pt) + BoT-SORT + RTSP实时流
模型: best11s896x672.pt  imgsz=(896,672)

三线程各自独立循环，非阻塞队列操作:
  采集线程: cap.read() → 录制 → 共享帧 → put_nowait提交推理(满则丢帧)
  推理线程: get(小超时)取帧 → GPU推理 → put_nowait放结果(满则替换旧结果)
  主线程:   get(小超时)取结果 → 后处理/追踪/计数 → HUD → 显示

丢帧检测: 采集丢帧即时告警 + 每10秒输出丢帧率统计

"""

import os
import sys
import time
import argparse
import threading
import logging
import signal
from datetime import datetime
from collections import deque
from queue import Queue, Empty, Full

import cv2
import math
import av
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class Config:
    rtsp_url = "rtsp://admin:@192.168.7.128:554/1/1"
    # rtsp_url = "rtsp://127.0.0.1:8554/my_stream"
    # rtsp_url = "src/2.mp4"
    model_path = "src/best.pt"
    tracker_yaml = "botsort.yaml"
    imgsz_w = 1280
    imgsz_h = 960
    save_dir = "D:/video"
    segment_duration = 600
    conf_threshold = 0.25
    iou_threshold = 0.6
    line_top_ratio = 50     # 斜线顶端x占宽度比例(%), 从(width*50%, 0)开始
    line_bot_ratio = 90     # 斜线底端x占宽度比例(%), 到(width*80%, height)结束
    device = "0"
    web_port = 5000
    record_mode = "qian"  # "qian"=采集线程录制, "hou"=主线程录制


class WebServer:
    def __init__(self, port=5000):
        self.port = port
        self.running = False
        self.frame_lock = threading.Lock()
        self.current_frame = None
        self.server_thread = None
        self.httpd = None

    def start(self):
        self.running = True
        self.server_thread = threading.Thread(target=self._run, daemon=True)
        self.server_thread.start()

    def stop(self):
        self.running = False
        if self.httpd:
            self.httpd.shutdown()
        if self.server_thread:
            self.server_thread.join(timeout=3)

    def update_frame(self, frame):
        with self.frame_lock:
            if frame is not None and frame.size > 0:
                self.current_frame = frame.copy()

    def _run(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        server_ref = self

        class MJPEGHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/video_feed':
                    self.request.settimeout(2.0)
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    while server_ref.running:
                        with server_ref.frame_lock:
                            frame = server_ref.current_frame
                        if frame is not None and frame.size > 0:
                            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            try:
                                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n')
                                self.wfile.write(f'Content-Length: {len(buf)}\r\n\r\n'.encode())
                                self.wfile.write(buf.tobytes())
                                self.wfile.write(b'\r\n')
                            except (ConnectionError, BrokenPipeError, OSError):
                                break
                        time.sleep(0.05)
                else:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.end_headers()
                    html = (
                        "<!DOCTYPE html><html><head>"
                        "<meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1.0'>"
                        "<title>猪只计数监控</title>"
                        "<style>body{background:#000;margin:0;text-align:center;}"
                        "h2{color:#0f0;padding:20px;}img{max-width:100%;height:auto;}</style>"
                        "</head><body><h2>猪只计数监控</h2>"
                        "<img src='/video_feed'></body></html>"
                    )
                    self.wfile.write(html.encode())

            def log_message(self, fmt, *args):
                pass

        try:
            self.httpd = ThreadingHTTPServer(('0.0.0.0', self.port), MJPEGHandler)
            logger.info(f"[WebServer] 已启动, 端口: {self.port}")
            self.httpd.serve_forever()
        except Exception as e:
            logger.warning(f"[WebServer] 启动失败: {e}")


class VideoRecorder:
    def __init__(self, save_dir, segment_duration, fps, width, height):
        self.save_dir = save_dir
        self.segment_duration = segment_duration
        self.fps = fps
        self.width = width
        self.height = height
        self.writer = None
        self.segment_start = time.time()
        self.current_filename = ""
        os.makedirs(save_dir, exist_ok=True)

    def write(self, frame):
        now = time.time()
        if self.writer is None or (now - self.segment_start) >= self.segment_duration:
            self._close()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_filename = os.path.join(self.save_dir, f"{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.current_filename, fourcc, self.fps,
                                          (self.width, self.height))
            self.segment_start = now
            logger.info(f"[录制] 开始: {self.current_filename}")

        if self.writer and self.writer.isOpened():
            self.writer.write(frame)

    def _close(self):
        if self.writer:
            self.writer.release()
            self.writer = None
            if self.current_filename:
                logger.info(f"[录制] 完成: {self.current_filename}")

    def release(self):
        self._close()



class FFmpegStream:
    """PyAV拉取RTSP流, 内部线程读帧到Queue
    无子进程/无管道, ffmpeg绑定在进程内直接运行, 零拷贝取帧
    """

    def __init__(self, url, width, height, queue_size=30):
        self.url = url
        self.width = width
        self.height = height
        self.queue = Queue(maxsize=queue_size)
        self.container = None
        self.thread = None
        self.running = False
        self.fail_count = 0

    def _open_container(self):
        try:
            options = {
                'rtsp_transport': 'tcp',
                'stimeout': '5000000',
                'fflags': 'nobuffer',
                'flags': 'low_delay',
                'reorder_queue_size': '0',
            }
            self.container = av.open(self.url, options=options, timeout=10.0)
            self.container.streams.video[0].thread_type = 'AUTO'
            logger.info(f"[PyAV] 连接成功: {self.url}")
            return True
        except Exception as e:
            logger.error(f"[PyAV] 连接失败: {e}")
            self.container = None
            return False

    def _read_worker(self):
        while self.running:
            if self.container is None:
                if not self._open_container():
                    time.sleep(1)
                    continue
            try:
                for packet in self.container.demux(video=0):
                    if not self.running:
                        break
                    for frame in packet.decode():
                        if not self.running:
                            break
                        img = frame.to_ndarray(format='bgr24')
                        try:
                            self.queue.put_nowait(img)
                        except Full:
                            pass
            except Exception as e:
                logger.warning(f"[PyAV] 读帧异常: {e}")
                self.fail_count += 1
            finally:
                if self.container is not None:
                    try:
                        self.container.close()
                    except Exception:
                        pass
                    self.container = None
            if self.running:
                time.sleep(2 if self.fail_count >= 10 else 0.1)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._read_worker, daemon=True)
        self.thread.start()
        # 等待第一帧就绪，避免 capture_worker 在首帧到达前误触发重连
        t0 = time.time()
        while time.time() - t0 < 5.0:
            ok, _ = self.read(timeout=0.5)
            if ok:
                break
        logger.info(f"[PyAV] 启动拉流: {self.url}")

    def read(self, timeout=0.05):
        try:
            return True, self.queue.get(timeout=timeout)
        except Empty:
            return False, None

    def stop(self):
        self.running = False
        if self.container is not None:
            try:
                self.container.close()
            except Exception:
                pass
            self.container = None
        if self.thread is not None:
            self.thread.join(timeout=3)

def connect_rtsp(url, retries=3):
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
        'rtsp_transport;tcp|stimeout;5000000|fflags;nobuffer|'
        'flags;low_delay|max_delay;0|reorder_queue_size;0'
    )
    cap = None
    for retry in range(retries):
        if retry > 0:
            logger.info(f"[系统] 重试连接 ({retry + 1}/{retries})...")
            time.sleep(2)
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            ret, test = cap.read()
            if ret and test is not None and test.size > 0:
                logger.info(f"[系统] RTSP连接成功 ({test.shape[1]}x{test.shape[0]})")
                return cap
            cap.release()
        cap = None
    return None


def main():
    parser = argparse.ArgumentParser(description='猪只计数监控系统 (YOLO11s+BoT-SORT)')
    parser.add_argument('--rtsp', type=str, default=Config.rtsp_url, help='RTSP流地址')
    parser.add_argument('--model', type=str, default=Config.model_path, help='模型路径 (.pt)')
    parser.add_argument('--tracker', type=str, default=Config.tracker_yaml, help='追踪器配置yaml')
    parser.add_argument('--imgsz-w', type=int, default=Config.imgsz_w, help='模型输入宽度')
    parser.add_argument('--imgsz-h', type=int, default=Config.imgsz_h, help='模型输入高度')
    parser.add_argument('--save', type=str, default=Config.save_dir, help='视频保存目录')
    parser.add_argument('--duration', type=int, default=Config.segment_duration, help='视频切片时长(秒)')
    parser.add_argument('--conf', type=float, default=Config.conf_threshold, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=Config.iou_threshold, help='NMS IoU阈值')
    parser.add_argument('--device', type=str, default=Config.device, help='推理设备 (0/cpu)')
    parser.add_argument('--web-port', type=int, default=Config.web_port, help='Web服务端口')
    args = parser.parse_args()

    cfg = Config()
    cfg.rtsp_url = args.rtsp
    cfg.model_path = args.model
    cfg.tracker_yaml = args.tracker
    cfg.imgsz_w = args.imgsz_w
    cfg.imgsz_h = args.imgsz_h
    cfg.save_dir = args.save
    cfg.segment_duration = args.duration
    cfg.conf_threshold = args.conf
    cfg.iou_threshold = args.iou
    cfg.device = args.device
    cfg.web_port = args.web_port

    imgsz = [cfg.imgsz_w, cfg.imgsz_h]

    print("=" * 40)
    print("  猪只计数监控系统 (YOLO11s + BoT-SORT)")
    print("=" * 40)

    print("[模型] 加载中...")
    model = YOLO(cfg.model_path)
    print("[模型] 加载完成")

    print("[模型] CUDA热身中...")
    dummy = np.zeros((960, 1280, 3), dtype=np.uint8)
    model.track(source=dummy, imgsz=imgsz, conf=cfg.conf_threshold,
                iou=cfg.iou_threshold, tracker=cfg.tracker_yaml,
                persist=True, device=cfg.device, verbose=False)
    print("[模型] CUDA热身完成")

    is_rtsp = cfg.rtsp_url.startswith("rtsp://")

    if is_rtsp:
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
            'rtsp_transport;tcp|stimeout;5000000|fflags;nobuffer|'
            'flags;low_delay|max_delay;0|reorder_queue_size;0'
        )
        probe_cap = cv2.VideoCapture(cfg.rtsp_url, cv2.CAP_FFMPEG)
        if not probe_cap.isOpened():
            logger.error("[错误] 无法连接RTSP源")
            return 1
        fps = int(probe_cap.get(cv2.CAP_PROP_FPS))
        width = int(probe_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(probe_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0: fps = 30
        ret, test = probe_cap.read()
        if ret and test is not None and test.size > 0:
            height, width = test.shape[:2]
        probe_cap.release()

        ffmpeg_stream = FFmpegStream(cfg.rtsp_url, width, height, queue_size=30)
        ffmpeg_stream.start()
        time.sleep(1)
        cap = None
    else:
        cap = cv2.VideoCapture(cfg.rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.error("[错误] 无法打开视频文件")
            return 1
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0: fps = 30
        if width <= 0 or height <= 0:
            ret, test = cap.read()
            if ret and test is not None and test.size > 0:
                height, width = test.shape[:2]
            else:
                width, height = 1280, 960
        ffmpeg_stream = None


    line_x1 = int(width * cfg.line_top_ratio / 100.0)  # 斜线顶端x
    line_x2 = int(width * cfg.line_bot_ratio / 100.0)  # 斜线底端x
    line_dx = line_x2 - line_x1  # 斜线x方向增量

    def line_x_at(cy):
        """给定y坐标, 返回斜线在该y处的x坐标"""
        return line_x1 + line_dx * cy / height

    # ====== 三线程异步流水线 ======
    frame_lock = threading.Lock()
    latest_frame = None

    infer_queue = Queue(maxsize=3)
    result_queue = Queue(maxsize=3)
    global_quit = False

    # ====== 丢帧统计 ======
    cap_count = 0
    infer_count = 0
    display_count = 0
    cap_drop_count = 0
    result_drop_count = 0
    stats_lock = threading.Lock()
    stats_start_time = time.time()

    # ---------- 推理线程 ----------
    def infer_worker():
        nonlocal infer_count, result_drop_count
        while not global_quit:
            try:
                item = infer_queue.get(timeout=0.1)
            except Empty:
                continue
            if item is None:
                break
            frm, t_submit = item
            try:
                results = model.track(
                    source=frm,
                    imgsz=imgsz,
                    conf=cfg.conf_threshold,
                    iou=cfg.iou_threshold,
                    tracker=cfg.tracker_yaml,
                    persist=True,
                    device=cfg.device,
                    verbose=False,
                )
                with stats_lock:
                    infer_count += 1
                try:
                    result_queue.put_nowait((results, t_submit))
                except Full:
                    with stats_lock:
                        result_drop_count += 1
                    logger.warning(f"[推理] 结果队列满, 丢弃结果 (累计丢弃 {result_drop_count})")
            except Exception as e:
                logger.error(f"[推理] 异常: {e}")

    # ---------- 采集线程 ----------
    cap_interval = 1.0 / fps if fps > 0 else 1.0 / 30

    def capture_worker():
        nonlocal latest_frame, global_quit, cap, cap_count, cap_drop_count
        local_recorder = VideoRecorder(cfg.save_dir, cfg.segment_duration, fps, width, height) if cfg.record_mode == "qian" else None
        rtsp_fail_count = 0
        next_read_time = time.perf_counter()
        while not global_quit:
            if is_rtsp:
                ret, frame = ffmpeg_stream.read(timeout=0.01)
            else:
                now = time.perf_counter()
                wait = next_read_time - now
                if wait > 0:
                    time.sleep(wait)
                next_read_time = time.perf_counter() + cap_interval
                ret = cap.grab()
                if ret:
                    ret, frame = cap.retrieve()
                else:
                    frame = None
            if not ret or frame is None or frame.size == 0:
                rtsp_fail_count += 1
                if rtsp_fail_count >= 30:
                    logger.warning("[采集] 连接中断, 尝试重连...")
                    if is_rtsp:
                        ffmpeg_stream.stop()
                        time.sleep(2)
                        ffmpeg_stream.start()
                        time.sleep(1)
                    else:
                        cap.release()
                        cap = cv2.VideoCapture(cfg.rtsp_url, cv2.CAP_FFMPEG)
                    rtsp_fail_count = 0
                    logger.info("[采集] 重连成功")
                continue
            rtsp_fail_count = 0
            with stats_lock:
                cap_count += 1
            if local_recorder is not None:
                local_recorder.write(frame)
            with frame_lock:
                latest_frame = frame
            try:
                infer_queue.put_nowait((frame, time.time()))
            except Full:
                with stats_lock:
                    cap_drop_count += 1
        if local_recorder is not None:
            local_recorder.release()

    # 启动WebServer
    web_server = WebServer(cfg.web_port)
    web_server.start()
    print(f"[系统] Web服务: http://0.0.0.0:{cfg.web_port}")

    # 启动采集和推理线程
    t_infer = threading.Thread(target=infer_worker, daemon=True)
    t_capture = threading.Thread(target=capture_worker, daemon=True)
    t_infer.start()
    t_capture.start()
    print("[系统] 三线程异步流水线: 采集 | 推理 | 显示")

    # ====== 计数状态 ======
    C = 0
    last_C_val = C
    last_C_change_time = time.time()
    crossed_ids = set()
    prev_positions = {}
    trails = {}
    unique_track_ids = set()
    pig_count = 0
    tracks = []

    # ---- ID缝合参数 ----
    STITCH_ANGLE_THRESH = math.radians(30)   # 方向夹角阈值(±30°)
    STITCH_TIME_MAX = 3                    # 丢失最长时间(秒)
    STITCH_DIST_MAX = 300                    # 预估位置最大距离(px)
    STITCH_SPEED_RATIO_MAX = 3.0            # 速度变化比上限
    STITCH_OCCUPY_DIST = 80                  # 占据检查距离(px)
    lost_tracks = {}                         # 丢失轨迹 {id: {cx, cy, vx, vy, angle, crossed, lost_time}}
    stitch_count = 0                         # 缝合成功次数

    # ---- 清零辅助函数 ----
    def _reset_counter():
        nonlocal C, last_C_val, last_C_change_time, crossed_ids, prev_positions, trails, lost_tracks, stitch_count
        C = 0
        crossed_ids.clear()
        prev_positions.clear()
        trails.clear()
        lost_tracks.clear()
        stitch_count = 0
        last_C_val = 0
        last_C_change_time = time.time()

    # hou模式录制器（主线程录制含HUD帧）
    hou_recorder = VideoRecorder(cfg.save_dir, cfg.segment_duration, fps, width, height) if cfg.record_mode == "hou" else None

    stat_infer_count = 0
    sum_preprocess_ms = 0.0
    sum_infer_only_ms = 0.0
    sum_postprocess_ms = 0.0

    last_stats_time = time.time()

    def signal_handler(sig, frame_sig):
        nonlocal global_quit
        global_quit = True

    signal.signal(signal.SIGINT, signal_handler)
    print("[系统] 运行中... (Q键/Ctrl+C 退出)")

    # ====== 主线程显示循环 ======
    while not global_quit:
        try:
            latest_results, latest_t_submit = result_queue.get(timeout=0.05)
        except Empty:
            pass
        else:
            display_count += 1
            tracks = []
            if latest_results and len(latest_results) > 0:
                r = latest_results[0]
                if r.boxes is not None and len(r.boxes) > 0 and r.boxes.id is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        track_id = int(box.id[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        tracks.append({
                            'id': track_id, 'cx': cx, 'cy': cy, 'conf': conf,
                        })
                spd = r.speed
                sum_preprocess_ms += spd.get('preprocess', 0)
                sum_infer_only_ms += spd.get('inference', 0)
                sum_postprocess_ms += spd.get('postprocess', 0)
                stat_infer_count += 1

            pig_count = len(tracks)
            for t in tracks:
                unique_track_ids.add(t['id'])

            # ---- ID缝合（速度+方向夹角匹配） ----
            new_tracks = [t for t in tracks
                          if t['id'] not in prev_positions and t['id'] not in lost_tracks]

            for nt in new_tracks:
                best_old_id = None
                best_match_dist = STITCH_DIST_MAX
                best_angle_diff = 0.0

                for old_id, info in lost_tracks.items():
                    dt = time.time() - info['lost_time']
                    if dt > STITCH_TIME_MAX:
                        continue

                    angle_diff = 0.0  # 默认方向差异为0

                    # 1. 速度预估位置距离检查
                    pred_cx = info['cx'] + info['vx'] * dt * fps
                    pred_cy = info['cy'] + info['vy'] * dt * fps
                    dist = math.hypot(nt['cx'] - pred_cx, nt['cy'] - pred_cy)
                    if dist > STITCH_DIST_MAX:
                        continue

                    # 2. 运动方向夹角检查（距离>5px时才有统计意义）
                    if dist > 5:
                        move_dx = nt['cx'] - info['cx']
                        move_dy = nt['cy'] - info['cy']
                        move_angle = math.atan2(move_dy, move_dx)
                        angle_diff = abs(move_angle - info['angle'])
                        if angle_diff > math.pi:
                            angle_diff = 2 * math.pi - angle_diff
                        if angle_diff > STITCH_ANGLE_THRESH:
                            continue

                    # 3. 速度大小一致性检查
                    if dist > 5 and dt > 0.05:
                        new_speed = dist / dt
                        old_speed = math.hypot(info['vx'], info['vy'])
                        if old_speed > 1.0:
                            speed_ratio = new_speed / old_speed
                            if speed_ratio > STITCH_SPEED_RATIO_MAX or speed_ratio < 1.0 / STITCH_SPEED_RATIO_MAX:
                                continue

                    # 4. 占据检查：预估位置附近有其他活跃track则拒绝
                    occupied = False
                    for at in tracks:
                        if at['id'] == nt['id']:
                            continue
                        if math.hypot(at['cx'] - pred_cx, at['cy'] - pred_cy) < STITCH_OCCUPY_DIST:
                            occupied = True
                            break
                    if occupied:
                        continue

                    if dist < best_match_dist:
                        best_match_dist = dist
                        best_old_id = old_id
                        best_angle_diff = angle_diff

                if best_old_id is not None:
                    info = lost_tracks[best_old_id]
                    nt['id'] = best_old_id
                    # 恢复旧ID的上一帧位置，使越线检测能正确判断
                    prev_positions[best_old_id] = (info['cx'], info['cy'])
                    # 继承越线状态
                    if info['crossed']:
                        crossed_ids.add(best_old_id)
                    elif best_old_id not in crossed_ids:
                        prev_side = info['cx'] - line_x_at(info['cy'])
                        cur_side = nt['cx'] - line_x_at(nt['cy'])
                        if prev_side < 0 and cur_side >= 0:
                            C += 1
                            crossed_ids.add(best_old_id)
                            prev_positions[best_old_id] = (nt['cx'], nt['cy'])
                            logger.info(f"[缝合计数] #{best_old_id} 左→右 (+1) C:{C}")
                        elif prev_side > 0 and cur_side <= 0:
                            C -= 1
                            crossed_ids.add(best_old_id)
                            prev_positions[best_old_id] = (nt['cx'], nt['cy'])
                            logger.info(f"[缝合计数] #{best_old_id} 右→左 (-1) C:{C}")
                    del lost_tracks[best_old_id]
                    stitch_count += 1
                    angle_str = f"夹角{math.degrees(best_angle_diff):.1f}°" if best_match_dist > 5 else "近距离"
                    logger.info(f"[ID缝合] #{best_old_id} 恢复 (距离{best_match_dist:.0f}px {angle_str})")

            # ---- 轨迹恢复：丢失后同ID重现，恢复prev_positions ----
            for t in tracks:
                tid = t['id']
                if tid in lost_tracks and tid not in prev_positions:
                    info = lost_tracks.pop(tid)
                    prev_positions[tid] = (info['cx'], info['cy'])
                    if info['crossed']:
                        crossed_ids.add(tid)
                    elif tid not in crossed_ids:
                        cur = next((t for t in tracks if t['id'] == tid), None)
                        if cur:
                            prev_side = info['cx'] - line_x_at(info['cy'])
                            cur_side = cur['cx'] - line_x_at(cur['cy'])
                            if prev_side < 0 and cur_side >= 0:
                                C += 1
                                crossed_ids.add(tid)
                                prev_positions[tid] = (cur['cx'], cur['cy'])
                                logger.info(f"[恢复计数] #{tid} 左→右 (+1) C:{C}")
                            elif prev_side > 0 and cur_side <= 0:
                                C -= 1
                                crossed_ids.add(tid)
                                prev_positions[tid] = (cur['cx'], cur['cy'])
                                logger.info(f"[恢复计数] #{tid} 右→左 (-1) C:{C}")

            # ---- 越线计数 ----
            for t in tracks:
                tid = t['id']
                cx, cy = t['cx'], t['cy']

                if tid in prev_positions:
                    prev_cx, prev_cy = prev_positions[tid]
                    prev_side = prev_cx - line_x_at(prev_cy)
                    cur_side = cx - line_x_at(cy)
                    if prev_side < 0 and cur_side >= 0:
                        C += 1
                        logger.info(f"[计数] #{tid} 左→右 (+1) C:{C}")
                    elif prev_side > 0 and cur_side <= 0:
                        C -= 1
                        logger.info(f"[计数] #{tid} 右→左 (-1) C:{C}")

                prev_positions[tid] = (cx, cy)

                if tid not in trails:
                    trails[tid] = deque(maxlen=60)
                trails[tid].append((cx, t['cy']))

            # 清理丢失轨迹：从prev_positions移入lost_tracks
            active_ids = {t['id'] for t in tracks}
            for tid in list(prev_positions.keys()):
                if tid not in active_ids:
                    if tid not in lost_tracks:
                        vx, vy = 0.0, 0.0
                        if tid in trails and len(trails[tid]) >= 5:
                            pts = list(trails[tid])
                            recent = pts[-5:]
                            if len(recent) >= 2:
                                vx = (recent[-1][0] - recent[0][0]) / (len(recent) - 1)
                                vy = (recent[-1][1] - recent[0][1]) / (len(recent) - 1)
                        angle = math.atan2(vy, vx) if math.hypot(vx, vy) > 1e-6 else 0.0
                        lost_tracks[tid] = {
                            'cx': prev_positions[tid][0],
                            'cy': prev_positions[tid][1],
                            'vx': vx, 'vy': vy,
                            'angle': angle,
                            'crossed': tid in crossed_ids,
                            'lost_time': time.time(),
                        }
                    del prev_positions[tid]

            # 清理超时的lost_tracks
            now_lt = time.time()
            for tid in list(lost_tracks.keys()):
                if now_lt - lost_tracks[tid]['lost_time'] > STITCH_TIME_MAX:
                    del lost_tracks[tid]

            # ---- C值30秒不变则清零 ----
            if C != last_C_val:
                last_C_val = C
                last_C_change_time = time.time()
            elif C != 0 and time.time() - last_C_change_time >= 30:
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open("log.txt", "a", encoding="utf-8") as f:
                    f.write(f"时间：{now_str}, 归零方式：30秒不变, C值：{C}\n")
                _reset_counter()
                logger.info("[清零] C 30秒不变，已归零")

        # HUD绘制
        with frame_lock:
            frame = latest_frame

        if frame is not None and frame.size > 0:
            cv2.line(frame, (line_x1, 0), (line_x2, height), (0, 255, 0), 2)

            for t in tracks:
                tid = t['id']
                if tid in trails:
                    trail = trails[tid]
                    pts = list(trail)
                    for i in range(1, len(pts)):
                        alpha = i / len(pts)
                        green = int(100 + 155 * alpha)
                        cv2.line(frame, pts[i - 1], pts[i], (0, green, 0), 2)

                cx, cy = t['cx'], t['cy']
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(tid), (cx + 8, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            cv2.putText(frame, f"C: {C}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 3)
            cv2.putText(frame, f"Pigs: {pig_count}", (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if hou_recorder is not None:
                hou_recorder.write(frame)

            display_frame = cv2.resize(frame, (int(width * 0.7), int(height * 0.7)))
            web_server.update_frame(display_frame)

            cv2.imshow("猪只计数监控", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                global_quit = True

        # 性能统计
        if stat_infer_count >= 30:
            avg_pre = sum_preprocess_ms / stat_infer_count
            avg_inf = sum_infer_only_ms / stat_infer_count
            avg_post = sum_postprocess_ms / stat_infer_count
            avg_total = avg_pre + avg_inf + avg_post
            infer_fps = 1000 / avg_total if avg_total > 0 else 0
            with stats_lock:
                cc = cap_count
                ic = infer_count
                dc = display_count
            print(f"[性能] pre:{avg_pre:.1f}ms infer:{avg_inf:.1f}ms post:{avg_post:.1f}ms "
                  f"total:{avg_total:.1f}ms ({infer_fps:.0f}fps) "
                  f"cap:{cc} infer:{ic} disp:{dc} "
                  f"pigs:{pig_count} tracks:{len(unique_track_ids)} C:{C} stitch:{stitch_count}")
            stat_infer_count = 0
            sum_preprocess_ms = 0.0
            sum_infer_only_ms = 0.0
            sum_postprocess_ms = 0.0

        # 每10秒输出丢帧率统计
        now_t = time.time()
        if now_t - last_stats_time >= 10.0:
            elapsed = now_t - stats_start_time
            with stats_lock:
                cc = cap_count
                ic = infer_count
                dc = display_count
                cd = cap_drop_count
                rd = result_drop_count
            if cc > 0:
                total_drop = cd + rd
                drop_rate = total_drop / cc * 100
                print(f"[丢帧率] {elapsed:.0f}s | 采集:{cc} 推理:{ic} 显示:{dc} | "
                      f"采集丢帧:{cd} 结果丢弃:{rd} 总丢帧:{total_drop}({drop_rate:.1f}%)")
            last_stats_time = now_t

    # ====== 清理 ======
    global_quit = True
    cv2.destroyAllWindows()
    for _ in range(30):
        cv2.waitKey(1)

    try:
        infer_queue.put_nowait(None)
    except Full:
        pass

    t_infer.join(timeout=3)
    t_capture.join(timeout=3)

    if cap is not None:
        cap.release()
    if ffmpeg_stream is not None:
        ffmpeg_stream.stop()
    if hou_recorder is not None:
        hou_recorder.release()
    web_server.stop()

    elapsed = time.time() - stats_start_time
    print(f"\n[结果] 最终计数 C:{C} 轨迹总数:{len(unique_track_ids)} ID缝合:{stitch_count}")
    print(f"[统计] 运行{elapsed:.0f}s | 采集:{cap_count} 推理:{infer_count} 显示:{display_count} | "
          f"采集丢帧:{cap_drop_count} 结果丢弃:{result_drop_count}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
