#!/usr/bin/env python3
"""抽取视频指定帧，用模型推理并画框保存"""

import cv2
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description='抽取指定帧并用模型推理画框')
    parser.add_argument('--video', default='src/2.mp4', help='视频路径')
    parser.add_argument('--model', default='src/best.pt', help='模型路径')
    parser.add_argument('--frame', type=int, default=2274, help='目标帧号(0起)')
    parser.add_argument('--output', default=None, help='输出图片路径(默认自动生成)')
    parser.add_argument('--conf', type=float, default=0.2, help='置信度阈值')
    parser.add_argument('--device', default='0', help='推理设备')
    args = parser.parse_args()

    # 打开视频
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {args.video}")
        return 1

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[视频] 总帧数: {total}, 目标帧: {args.frame}")

    if args.frame >= total:
        print(f"[错误] 帧号 {args.frame} 超出总帧数 {total}")
        cap.release()
        return 1

    # 跳转到目标帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None or frame.size == 0:
        print(f"[错误] 无法读取第 {args.frame} 帧")
        return 1

    h, w = frame.shape[:2]
    print(f"[帧] 第 {args.frame} 帧: {w}x{h}")

    # 加载模型推理
    model = YOLO(args.model)
    results = model.predict(source=frame, conf=args.conf, device=args.device, verbose=False)[0]
    print(f"[推理] 检测到 {len(results.boxes)} 个目标")

    # 画框
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 保存
    output = args.output or f"frame_{args.frame}.jpg"
    cv2.imwrite(output, frame)
    print(f"[保存] {output}")
    return 0


if __name__ == '__main__':
    exit(main())
