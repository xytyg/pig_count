#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
猪只计数监控系统 (三线程异步流水线)
基于 YOLO26n (.pt) + BoT-SORT + RTSP实时流
模型: best.pt  imgsz=1280 rect=True 

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
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk


# ---- UI与管线通信 (由UI框架设置, None/默认值=无UI) ----
class _UIControl:
    stop_event = threading.Event()
    reset_event = threading.Event()
    c_log_queue = None      # Queue, 管线→UI的归零日志
    frame_queue = None      # Queue, 管线→UI的视频帧
    stats_queue = None      # Queue, 管线→UI的统计数据
    status_msg = ""         # 最新状态消息（跨线程传递）

_ui_ctl = _UIControl()

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class Config:
    rtsp_url = "rtsp://admin:@192.168.7.128:554/1/1"
    # rtsp_url = "src/2.mp4"
    model_path = "src/best.pt"
    tracker_yaml = "botsort.yaml"
    imgsz_w = 1280
    imgsz_h = 960
    save_dir = "D:/video"
    segment_duration = 600
    conf_threshold = 0.25
    iou_threshold = 0.6
    # 折线检测线 - 三个点 (水平%, 垂直%)
    poly_p1 = (0, 85)    # 点1
    poly_p2 = (60, 75)   # 点2
    poly_p3 = (100, 0)  # 点3
    device = "0"
    web_port = 5000
    record_mode = "qian"  # "qian"=采集线程录制, "hou"=主线程录制


class CountingUI:
    """猪只计数监控系统UI界面"""

    def __init__(self, config, on_start_callback, on_stop_callback, on_reset_callback, root=None):
        self.config = config
        self.on_start = on_start_callback
        self.on_stop = on_stop_callback
        self.on_reset = on_reset_callback

        # 使用外部传入的root窗口或创建新的
        if root is not None:
            self.root = root
            self.own_root = False
        else:
            self.root = tk.Tk()
            self.root.title("猪只计数监控系统")
            self.root.state('zoomed')  # 默认最大化
            self.own_root = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 状态变量
        self.is_running = True   # UI窗口是否活跃
        self.window_closed = False  # 窗口是否被关闭（区别于停止运行）
        self.current_frame = None
        self.C_value = 0
        self.pig_count = 0
        self.fps = 0
        self.drop_rate = 0.0
        self._frame_skip = 0

        # 创建UI组件
        self._setup_style()
        self._create_widgets()

    def _setup_style(self):
        """配置简洁灰色主题"""
        self.C = {  # 配色字典
            'bg': '#f3f4f6',           # 窗口背景 - 浅灰
            'card': '#ffffff',          # 卡片背景 - 纯白
            'primary': '#374151',       # 主色 - 深灰（标题、边框）
            'primary_bg': '#f3f4f6',    # 主色背景 - 浅灰
            'text': '#1e293b',          # 正文 - 深灰
            'text_muted': '#9ca3af',    # 辅助文字 - 浅灰
            'border': '#e5e7eb',        # 边框 - 灰色
            'canvas_bg': '#1e293b',     # 视频区背景
            'status_bg': '#f9fafb',     # 状态栏背景 - 极浅灰
            'status_fg': '#4b5563',     # 状态栏文字 - 中灰
        }
        self.root.configure(bg=self.C['bg'])

    def _create_widgets(self):
        """创建UI组件"""
        # 主框架 - pack左右布局（右面板按高定宽，左面板吃余宽）
        main_frame = tk.Frame(self.root, bg=self.C['card'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 右面板先pack（按高定宽），左面板后pack（吃剩余空间）
        self.right_panel = tk.Frame(main_frame, bg=self.C['card'])
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        self.right_panel.pack_propagate(False)

        # 左侧控制面板
        left_panel = tk.Frame(main_frame, bg=self.C['card'],
                              highlightbackground=self.C['border'], highlightthickness=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 上半部分 - 控制区域（加宽松间距）
        upper_frame = tk.Frame(left_panel, bg=self.C['card'])
        upper_frame.pack(fill=tk.X, pady=(10, 0), padx=4)

        # 视频源设置
        source_frame = tk.LabelFrame(upper_frame, text="视频源", padx=8, pady=5,
                                     bg=self.C['card'], fg=self.C['primary'],
                                     font=("Arial", 9, "bold"),
                                     highlightbackground=self.C['border'], highlightthickness=1)
        source_frame.pack(fill=tk.X, padx=6, pady=4)

        rtsp_frame = tk.Frame(source_frame, bg=self.C['card'])
        rtsp_frame.pack(fill=tk.X, pady=(0, 3))
        self.rtsp_entry = ttk.Entry(rtsp_frame, width=28, font=("Arial", 9))
        self.rtsp_entry.insert(0, self.config.rtsp_url)
        self.rtsp_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.browse_file_btn = tk.Button(rtsp_frame, text="…", width=3, relief=tk.RAISED, borderwidth=1,
                                         bg=self.C['primary_bg'], fg=self.C['primary'], cursor='hand2',
                                         command=self._browse_local_file)
        self.browse_file_btn.pack(side=tk.LEFT, padx=(3, 0))
        # 根据初始内容决定浏览按钮显隐
        self._update_source_browse_btn()

        rtsp_btn_frame = tk.Frame(source_frame, bg=self.C['card'])
        rtsp_btn_frame.pack(fill=tk.X)
        tk.Button(rtsp_btn_frame, text="本地视频", width=10, relief=tk.FLAT,
                  bg=self.C['primary_bg'], fg=self.C['primary'], activebackground=self.C['primary'],
                  activeforeground='white', cursor='hand2',
                  command=lambda: (self.rtsp_entry.delete(0, tk.END), self.rtsp_entry.insert(0, "src/2.mp4"), self._update_source_browse_btn())).pack(side=tk.LEFT, padx=2)
        tk.Button(rtsp_btn_frame, text="摄像头", width=10, relief=tk.FLAT,
                  bg=self.C['primary_bg'], fg=self.C['primary'], activebackground=self.C['primary'],
                  activeforeground='white', cursor='hand2',
                  command=lambda: (self.rtsp_entry.delete(0, tk.END), self.rtsp_entry.insert(0, "rtsp://admin:@192.168.7.128:554/1/1"), self._update_source_browse_btn())).pack(side=tk.LEFT, padx=2)

        # 模型 & 计数线参数
        model_frame = tk.LabelFrame(upper_frame, text="模型 & 计数线", padx=8, pady=5,
                                    bg=self.C['card'], fg=self.C['primary'],
                                    font=("Arial", 9, "bold"),
                                    highlightbackground=self.C['border'], highlightthickness=1)
        model_frame.pack(fill=tk.X, padx=6, pady=4)

        conf_frame = tk.Frame(model_frame, bg=self.C['card'])
        conf_frame.pack(fill=tk.X, pady=(0, 2))
        tk.Label(conf_frame, text="置信度:", bg=self.C['card'], fg=self.C['text'],
                 font=("Arial", 9)).pack(side=tk.LEFT)
        self.conf_var = tk.DoubleVar(value=self.config.conf_threshold)
        self.conf_scale = ttk.Scale(conf_frame, from_=0.1, to=0.9, variable=self.conf_var)
        self.conf_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.conf_label = tk.Label(conf_frame, text=f"{self.config.conf_threshold:.2f}",
                                   bg=self.C['card'], fg=self.C['primary'], width=5, font=("Arial", 9))
        self.conf_label.pack(side=tk.RIGHT)
        self.conf_var.trace_add("write", self._on_conf_change)

        iou_frame = tk.Frame(model_frame, bg=self.C['card'])
        iou_frame.pack(fill=tk.X, pady=(0, 2))
        tk.Label(iou_frame, text="IoU:", bg=self.C['card'], fg=self.C['text'],
                 font=("Arial", 9)).pack(side=tk.LEFT)
        self.iou_var = tk.DoubleVar(value=self.config.iou_threshold)
        self.iou_scale = ttk.Scale(iou_frame, from_=0.1, to=0.9, variable=self.iou_var)
        self.iou_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.iou_label = tk.Label(iou_frame, text=f"{self.config.iou_threshold:.2f}",
                                  bg=self.C['card'], fg=self.C['primary'], width=5, font=("Arial", 9))
        self.iou_label.pack(side=tk.RIGHT)
        self.iou_var.trace_add("write", self._on_iou_change)

        # ====== 折线检测线（三个点） ======
        self._create_polyline_ui(model_frame)

        # ====== 录像设置 ======
        record_frame = tk.LabelFrame(left_panel, text="录像设置", padx=8, pady=5,
                                     bg=self.C['card'], fg=self.C['primary'],
                                     font=("Arial", 9, "bold"),
                                     highlightbackground=self.C['border'], highlightthickness=1)
        record_frame.pack(fill=tk.X, padx=6, pady=2)

        path_frame = tk.Frame(record_frame, bg=self.C['card'])
        path_frame.pack(fill=tk.X, pady=(0, 4))
        tk.Label(path_frame, text="保存路径:", bg=self.C['card'], fg=self.C['text'],
                 font=("Arial", 9)).pack(side=tk.LEFT)
        self.save_dir_entry = ttk.Entry(path_frame, width=24, font=("Arial", 9))
        self.save_dir_entry.insert(0, self.config.save_dir)
        self.save_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        self.browse_save_btn = tk.Button(path_frame, text="浏览", width=4, relief=tk.RAISED, borderwidth=1,
                                         bg=self.C['primary_bg'], fg=self.C['primary'], cursor='hand2',
                                         command=self._browse_save_dir)
        self.browse_save_btn.pack(side=tk.LEFT, padx=(3, 0))

        mode_frame = tk.Frame(record_frame, bg=self.C['card'])
        mode_frame.pack(fill=tk.X)
        tk.Label(mode_frame, text="保存时机:", bg=self.C['card'], fg=self.C['text'],
                 font=("Arial", 9)).pack(side=tk.LEFT)
        self.record_mode_var = tk.StringVar(value=self.config.record_mode)
        tk.Radiobutton(mode_frame, text="推理前", variable=self.record_mode_var,
                       value="qian", bg=self.C['card'], fg=self.C['text'],
                       selectcolor=self.C['primary_bg'], activebackground=self.C['card'],
                       font=("Arial", 9)).pack(side=tk.LEFT, padx=8)
        tk.Radiobutton(mode_frame, text="推理后", variable=self.record_mode_var,
                       value="hou", bg=self.C['card'], fg=self.C['text'],
                       selectcolor=self.C['primary_bg'], activebackground=self.C['card'],
                       font=("Arial", 9)).pack(side=tk.LEFT, padx=8)

        # ====== 控制按钮（居中，总宽度70%） ======
        btn_frame = tk.Frame(left_panel, bg=self.C['card'])
        btn_frame.pack(fill=tk.X, padx=6, pady=(20, 2))
        btn_frame.columnconfigure(0, weight=15)
        btn_frame.columnconfigure(1, weight=70)
        btn_frame.columnconfigure(2, weight=15)

        btn_inner = tk.Frame(btn_frame, bg=self.C['card'])
        btn_inner.grid(row=0, column=1, sticky='ew')
        btn_inner.columnconfigure((0, 1, 2), weight=1)

        btn_bg = '#f3f4f6'
        self.start_btn = tk.Button(btn_inner, text="开始", command=self._on_start,
                                   bg=btn_bg, fg='green', relief=tk.RAISED, borderwidth=1,
                                   activebackground='#e5e7eb', activeforeground='#006400',
                                   font=("Arial", 10, "bold"), cursor='hand2', padx=14, pady=4)
        self.start_btn.grid(row=0, column=0, sticky='ew', padx=5)

        self.stop_btn = tk.Button(btn_inner, text="停止", command=self._on_stop,
                                  state=tk.DISABLED,
                                  bg=btn_bg, fg='red', relief=tk.RAISED, borderwidth=1,
                                  activebackground='#e5e7eb', activeforeground='#8b0000',
                                  font=("Arial", 10, "bold"), cursor='hand2', padx=14, pady=4)
        self.stop_btn.grid(row=0, column=1, sticky='ew', padx=5)

        self.reset_btn = tk.Button(btn_inner, text="清零", command=self._on_reset,
                                   bg=btn_bg, fg='blue', relief=tk.RAISED, borderwidth=1,
                                   activebackground='#e5e7eb', activeforeground='#00008b',
                                   font=("Arial", 10, "bold"), cursor='hand2', padx=14, pady=4)
        self.reset_btn.grid(row=0, column=2, sticky='ew', padx=5)

        # ====== 底部历史计数（填满按钮以下全部空间） ======
        lower_frame = tk.Frame(left_panel, bg=self.C['card'])
        lower_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 0))
        btn_frame.pack_configure(pady=(20, 30))  # 按钮底部留一个按钮高度

        self.HIST_COLS = 10
        self.HIST_ROWS = 15  # 共150格，字体加大后仍能显示足够数据
        history_frame = tk.LabelFrame(lower_frame, text="计数历史", padx=4, pady=4,
                                      bg=self.C['card'], fg=self.C['primary'],
                                      font=("Arial", 9, "bold"),
                                      highlightbackground=self.C['border'], highlightthickness=1)
        history_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 2))

        self.history_grid = []
        for c in range(self.HIST_COLS):
            col_frame = tk.Frame(history_frame, bg=self.C['card'])
            col_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=0)
            col_labels = []
            for _ in range(self.HIST_ROWS):
                lbl = tk.Label(col_frame, text="", font=("Consolas", 10), width=5, anchor='w',
                               bg=self.C['card'], fg=self.C['text'])
                lbl.pack(fill=tk.BOTH, expand=True, pady=0)
                col_labels.append(lbl)
            self.history_grid.append(col_labels)

        self.right_panel.bind('<Configure>', self._on_video_resize)

        self.video_canvas = tk.Canvas(self.right_panel, bg=self.C['canvas_bg'],
                                      highlightbackground=self.C['primary'],
                                      highlightthickness=2, highlightcolor=self.C['primary'])
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # 底部状态栏 - 简洁风格
        status_bar = tk.Frame(self.root, height=30, bg=self.C['status_bg'],
                              highlightbackground=self.C['border'], highlightthickness=1)
        status_bar.pack(fill=tk.X, padx=10, pady=(0, 10))
        status_bar.pack_propagate(False)

        sb = self.C['status_bg']
        sf = self.C['status_fg']
        self.status_label = tk.Label(status_bar, text="就绪", bg=sb, fg=sf, font=("Arial", 9))
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.c_value_label = tk.Label(status_bar, text="C值: 0", bg=sb, fg=self.C['primary'],
                                      font=("Arial", 11, "bold"))
        self.c_value_label.pack(side=tk.LEFT, padx=15)

        self.fps_label = tk.Label(status_bar, text="FPS: 0.0", bg=sb, fg=sf, font=("Arial", 9))
        self.fps_label.pack(side=tk.LEFT, padx=15)

        self.drop_label = tk.Label(status_bar, text="采丢:0 结丢:0", bg=sb, fg=sf, font=("Arial", 9))
        self.drop_label.pack(side=tk.LEFT, padx=15)

        self.total_label = tk.Label(status_bar, text="总数: 0", bg=sb, fg=sf, font=("Arial", 9))
        self.total_label.pack(side=tk.LEFT, padx=15)

        self.status_msg_var = tk.StringVar(value="")
        self.status_msg_label = tk.Label(status_bar, textvariable=self.status_msg_var,
                                         bg=sb, fg=self.C['text_muted'], font=("Consolas", 9))
        self.status_msg_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.time_label = tk.Label(status_bar, text="", bg=sb, fg=sf, font=("Arial", 9))
        self.time_label.pack(side=tk.RIGHT, padx=10)

    def _on_conf_change(self, *args):
        """置信度阈值变化回调"""
        value = self.conf_var.get()
        self.conf_label.config(text=f"{value:.2f}")
        self.config.conf_threshold = value

    def _on_iou_change(self, *args):
        """IoU阈值变化回调"""
        value = self.iou_var.get()
        self.iou_label.config(text=f"{value:.2f}")
        self.config.iou_threshold = value

    def _create_polyline_ui(self, parent):
        """创建折线检测线三个点的输入控件（P1横坐标固定0，P3横坐标固定100）"""
        poly_frame = tk.Frame(parent, bg=self.C['card'])
        poly_frame.pack(fill=tk.X)
        tk.Label(poly_frame, text="检测线:", bg=self.C['card'], fg=self.C['text'],
                 font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 6))

        self.poly_vars = {}
        for pt_i in (1, 2, 3):
            pt_frame = tk.Frame(poly_frame, bg=self.C['card'])
            pt_frame.pack(side=tk.LEFT, padx=4)
            tk.Label(pt_frame, text=f"P{pt_i}", bg=self.C['card'], fg=self.C['primary'],
                     font=("Arial", 8, "bold")).pack(side=tk.LEFT)

            h_str = str(getattr(self.config, f'poly_p{pt_i}')[0])
            v_var = tk.StringVar(value=str(getattr(self.config, f'poly_p{pt_i}')[1]))

            if pt_i == 1:
                tk.Label(pt_frame, text="0", width=3, bg=self.C['card'],
                         fg=self.C['text_muted'], font=("Arial", 9)).pack(side=tk.LEFT)
                tk.Label(pt_frame, text="H", bg=self.C['card'], fg=self.C['text_muted'],
                         font=("Arial", 7)).pack(side=tk.LEFT)
                self.poly_vars[pt_i] = (tk.StringVar(value="0"), v_var)
            elif pt_i == 2:
                h_var = tk.StringVar(value=h_str)
                e_h = ttk.Entry(pt_frame, textvariable=h_var, width=3)
                e_h.pack(side=tk.LEFT)
                tk.Label(pt_frame, text="H", bg=self.C['card'], fg=self.C['text_muted'],
                         font=("Arial", 7)).pack(side=tk.LEFT)
                h_var.trace_add("write", lambda *_, i=pt_i: self._on_poly_change(i))
                self.poly_vars[pt_i] = (h_var, v_var)
            else:
                tk.Label(pt_frame, text="100", width=3, bg=self.C['card'],
                         fg=self.C['text_muted'], font=("Arial", 9)).pack(side=tk.LEFT)
                tk.Label(pt_frame, text="H", bg=self.C['card'], fg=self.C['text_muted'],
                         font=("Arial", 7)).pack(side=tk.LEFT)
                self.poly_vars[pt_i] = (tk.StringVar(value="100"), v_var)

            e_v = ttk.Entry(pt_frame, textvariable=v_var, width=3)
            e_v.pack(side=tk.LEFT, padx=(2, 0))
            tk.Label(pt_frame, text="V", bg=self.C['card'], fg=self.C['text_muted'],
                     font=("Arial", 7)).pack(side=tk.LEFT)
            v_var.trace_add("write", lambda *_, i=pt_i: self._on_poly_change(i))

    def _on_poly_change(self, pt_idx):
        """折线检测线参数变化回调"""
        try:
            h = int(self.poly_vars[pt_idx][0].get())
            v = int(self.poly_vars[pt_idx][1].get())
            h = max(0, min(100, h))
            v = max(0, min(100, v))
            setattr(self.config, f'poly_p{pt_idx}', (h, v))
        except (ValueError, AttributeError):
            pass

    def _on_start(self):
        """开始按钮回调"""
        self.config.rtsp_url = self.rtsp_entry.get()
        self.config.save_dir = self.save_dir_entry.get()
        self.config.record_mode = self.record_mode_var.get()
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="运行中...")
        if self.on_start:
            self.on_start()

    def _on_stop(self):
        """停止按钮回调 — 仅停止管线，不关闭窗口"""
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="已停止")
        if self.on_stop:
            self.on_stop()

    def _on_reset(self):
        """清零计数回调"""
        if self.on_reset:
            self.on_reset()

    def _on_video_resize(self, event):
        """视频面板尺寸变化时按4:3比例调整宽度"""
        if event.widget != self.right_panel:
            return
        if getattr(self, '_adjusting_video', False):
            return
        h = event.height
        if h > 50:
            target_w = int(h * 4 / 3)
            if abs(event.width - target_w) > 5:
                self._adjusting_video = True
                self.right_panel.config(width=target_w)
                self._adjusting_video = False

    def _browse_save_dir(self):
        """浏览选择录像保存目录"""
        path = filedialog.askdirectory(initialdir=self.config.save_dir, title="选择录像保存目录")
        if path:
            self.save_dir_entry.delete(0, tk.END)
            self.save_dir_entry.insert(0, path)
            self.config.save_dir = path

    def _browse_local_file(self):
        """浏览选择本地视频文件"""
        initial = os.path.dirname(self.rtsp_entry.get()) if os.path.isfile(self.rtsp_entry.get()) else "/"
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mkv *.mov"), ("所有文件", "*.*")],
            initialdir=initial,
        )
        if path:
            self.rtsp_entry.delete(0, tk.END)
            self.rtsp_entry.insert(0, path)
            self._update_source_browse_btn()

    def _update_source_browse_btn(self):
        """根据视频源类型启用/禁用文件浏览按钮（保留占位，避免布局跳动）"""
        url = self.rtsp_entry.get()
        if url.startswith("rtsp://"):
            self.browse_file_btn.config(state=tk.DISABLED, relief=tk.FLAT,
                                        bg=self.C['card'], fg=self.C['border'], cursor='arrow')
        else:
            self.browse_file_btn.config(state=tk.NORMAL, relief=tk.RAISED, borderwidth=1,
                                        bg=self.C['primary_bg'], fg=self.C['primary'], cursor='hand2')

    def update_frame(self, frame, C_value, pig_count, fps, drop_rate):
        """更新视频帧和状态"""
        self.current_frame = frame
        self.C_value = C_value
        self.pig_count = pig_count
        self.fps = fps
        self.drop_rate = drop_rate

        self._frame_skip += 1
        # 更新视频显示（隔帧渲染，降低PIL/tk开销）
        if frame is not None and self._frame_skip % 2 == 0:
            try:
                # 调整大小以适应画布
                canvas_width = self.video_canvas.winfo_width()
                canvas_height = self.video_canvas.winfo_height()

                if canvas_width > 10 and canvas_height > 10:
                    h, w = frame.shape[:2]
                    scale = min(canvas_width / w, canvas_height / h)
                    new_w, new_h = int(w * scale), int(h * scale)

                    frame = cv2.resize(frame, (new_w, new_h))

                    # 转换为RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img_tk = ImageTk.PhotoImage(image=img)

                    # 在画布上显示
                    self.video_canvas.delete("all")
                    self.video_canvas.create_image(
                        canvas_width // 2, canvas_height // 2,
                        image=img_tk, anchor=tk.CENTER
                    )
                    self.video_canvas.image = img_tk  # 保持引用
            except:
                pass

        # 更新状态标签（每帧更新）
        self.c_value_label.config(text=f"C值: {self.C_value}")
        self.fps_label.config(text=f"FPS: {self.fps:.1f}")

        # 更新时间
        self.time_label.config(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def update_history(self, history_data):
        """更新计数历史显示（左小右大，带圈序号）"""
        for col in self.history_grid:
            for lbl in col:
                lbl.config(text="")

        total = self.HIST_COLS * self.HIST_ROWS
        start_idx = max(0, len(history_data) - total)
        visible = history_data[start_idx:]
        for i, value in enumerate(visible):
            col = i // self.HIST_ROWS
            row = i % self.HIST_ROWS
            if 0 <= col < self.HIST_COLS and 0 <= row < self.HIST_ROWS:
                self.history_grid[col][row].config(text=str(value))

    def on_closing(self):
        """窗口关闭回调"""
        self.window_closed = True
        if self.on_stop:
            self.on_stop()
        if self.own_root:
            self.root.quit()
            self.root.destroy()

    def run(self):
        """运行UI主循环"""
        if self.own_root:
            self.root.mainloop()


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
                'stimeout': '500000',  # 0.5秒超时，让demux定期返回以便检查退出标志
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
        # 等待采集线程自然退出（stimeout=0.5s 让 demux 定期返回检查退出标志）
        if self.thread is not None:
            self.thread.join(timeout=3)
        # 线程已退出，它的 finally 块已安全关闭 container
        if self.container is not None:
            try:
                self.container.close()
            except Exception:
                pass
            self.container = None

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
    parser.add_argument('--no-ui', action='store_true', help='不显示UI界面')
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

    # 初始化UI控制队列
    _ui_ctl.frame_queue = Queue(maxsize=3)
    _ui_ctl.stats_queue = Queue(maxsize=3)

    # 视频源变量（延迟初始化，在 start_system() 中赋值）
    is_rtsp = False
    fps = 30
    width = 1280
    height = 960
    cap = None
    ffmpeg_stream = None

    def line_y_at(cx):
        """给定x坐标, 返回折线在该x处的y坐标（从cfg实时读取参数）"""
        p1x = int(width * cfg.poly_p1[0] / 100.0)
        p1y = int(height * cfg.poly_p1[1] / 100.0)
        p2x = int(width * cfg.poly_p2[0] / 100.0)
        p2y = int(height * cfg.poly_p2[1] / 100.0)
        p3x = int(width * cfg.poly_p3[0] / 100.0)
        p3y = int(height * cfg.poly_p3[1] / 100.0)
        if cx <= p2x:
            if p2x != p1x:
                return p1y + (p2y - p1y) * (cx - p1x) / (p2x - p1x)
            return p1y
        else:
            if p3x != p2x:
                return p2y + (p3y - p2y) * (cx - p2x) / (p3x - p2x)
            return p2y

    # ====== 三线程异步流水线 ======
    frame_lock = threading.Lock()
    latest_frame = None

    infer_queue = Queue(maxsize=100)
    result_queue = Queue(maxsize=1000)  # 只存小结果(无帧数据)，可设大防丢
    global_quit = False
    system_running = False  # 系统运行状态

    # ====== 帧缓存（帧ID索引，避免结果队列存大帧） ======
    frame_buffer = {}          # frame_id → frame副本
    frame_buffer_lock = threading.Lock()
    frame_id_counter = 0
    MAX_FRAMES_IN_BUFFER = 60  # 保留约2秒的帧

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
            frm, fid, t_submit = item  # (frame, frame_id, submit_time)
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
                    # 传帧ID而非帧副本，结果队列可设很大而不占内存
                    result_queue.put_nowait((results, fid, t_submit))
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
        nonlocal frame_id_counter
        local_recorder = VideoRecorder(cfg.save_dir, cfg.segment_duration, fps, width, height) if cfg.record_mode == "qian" else None
        rtsp_fail_count = 0
        next_read_time = time.perf_counter()
        while not global_quit and system_running:
            if is_rtsp:
                ret, frame = ffmpeg_stream.read(timeout=0.01)
            else:
                if cap is None:
                    time.sleep(0.1)
                    continue
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
                    _ui_ctl.status_msg = "连接中断, 尝试重连..."
                    if is_rtsp:
                        ffmpeg_stream.stop()
                        time.sleep(2)
                        ffmpeg_stream.start()
                        time.sleep(1)
                    else:
                        if cap is not None:
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
            # 分配帧ID并存入缓存（供主循环按ID取帧画HUD）
            with frame_buffer_lock:
                fid = frame_id_counter
                frame_id_counter += 1
                frame_buffer[fid] = frame.copy()
                while len(frame_buffer) > MAX_FRAMES_IN_BUFFER:
                    oldest = min(frame_buffer.keys())
                    del frame_buffer[oldest]
            with frame_lock:
                latest_frame = frame
            try:
                infer_queue.put_nowait((frame, fid, time.time()))
            except Full:
                with stats_lock:
                    cap_drop_count += 1
                logger.warning(f"[采集] 推理队列满, 丢弃帧 (累计丢弃 {cap_drop_count})")
                # 本地视频模式下等待队列有空间，避免疯狂丢帧
                if not is_rtsp:
                    try:
                        infer_queue.put((frame, fid, time.time()), timeout=0.5)
                    except Empty:
                        pass
        if local_recorder is not None:
            local_recorder.release()

    # 启动WebServer
    web_server = WebServer(cfg.web_port)
    web_server.start()
    print(f"[系统] Web服务: http://0.0.0.0:{cfg.web_port}")

    # 线程变量（延迟初始化）
    t_infer = None
    t_capture = None
    ffmpeg_stream = None
    cap = None
    local_recorder = None

    # 启动系统的函数
    def start_system():
        nonlocal t_infer, t_capture, ffmpeg_stream, cap, local_recorder, system_running
        nonlocal is_rtsp, fps, width, height
        nonlocal cap_count, infer_count, display_count, cap_drop_count, result_drop_count, stats_start_time
        nonlocal hou_recorder

        if system_running:
            logger.info("[系统] 系统已在运行中")
            return

        logger.info("[系统] 启动视频流和推理...")

        # 重置统计
        cap_count = 0
        infer_count = 0
        display_count = 0
        cap_drop_count = 0
        result_drop_count = 0
        stats_start_time = time.time()

        # 清空队列
        while not infer_queue.empty():
            try:
                infer_queue.get_nowait()
            except Empty:
                break
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except Empty:
                break

        # 检测视频源类型
        is_rtsp = cfg.rtsp_url.startswith("rtsp://")

        if is_rtsp:
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                'rtsp_transport;tcp|stimeout;2000000|fflags;nobuffer|'
                'flags;low_delay|max_delay;0|reorder_queue_size;0'
            )
            probe_cap = cv2.VideoCapture(cfg.rtsp_url, cv2.CAP_FFMPEG)
            if not probe_cap.isOpened():
                logger.error("[错误] 无法连接RTSP源")
                _ui_ctl.status_msg = "无法连接RTSP源"
                return
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
                _ui_ctl.status_msg = "无法打开视频文件"
                return
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

        # 标记系统运行（必须在启动线程前设置，避免采集线程因竞争条件直接退出）
        system_running = True

        # 启动采集和推理线程
        t_infer = threading.Thread(target=infer_worker, daemon=True)
        t_capture = threading.Thread(target=capture_worker, daemon=True)
        t_infer.start()
        t_capture.start()

        # 启动录制器
        if cfg.record_mode == "qian":
            local_recorder = VideoRecorder(cfg.save_dir, cfg.segment_duration, fps, width, height)
        elif cfg.record_mode == "hou":
            if hou_recorder is not None:
                hou_recorder.release()
            hou_recorder = VideoRecorder(cfg.save_dir, cfg.segment_duration, fps, width, height)

        logger.info("[系统] 视频流和推理已启动")

    # 停止系统的函数
    def stop_system():
        nonlocal t_infer, t_capture, ffmpeg_stream, cap, local_recorder, hou_recorder, system_running

        if not system_running:
            logger.info("[系统] 系统未在运行")
            return

        logger.info("[系统] 停止视频流和推理...")

        system_running = False
        # 不设置 global_quit = True，只停止系统运行

        # 先等待采集线程自然退出（system_running=False 使其退出循环）
        if t_capture:
            t_capture.join(timeout=5)

        # 采集线程已退出，安全释放资源
        if cap is not None:
            cap.release()
            cap = None
        if ffmpeg_stream is not None:
            ffmpeg_stream.stop()
            ffmpeg_stream = None

        # 停止推理线程
        try:
            infer_queue.put_nowait(None)
        except Full:
            pass
        if t_infer:
            t_infer.join(timeout=3)

        # 释放录制器
        if local_recorder is not None:
            local_recorder.release()
            local_recorder = None
        if hou_recorder is not None:
            hou_recorder.release()
            hou_recorder = None

        logger.info("[系统] 视频流和推理已停止")
        _ui_ctl.status_msg = f"已停止, 录像保存至: {cfg.save_dir}"

    # ====== 计数状态 ======
    C = 0
    last_C_val = C
    last_C_change_time = time.time()
    # ---- 清零辅助函数 ----
    history_data = []  # 存储历史计数数据

    def _reset_counter():
        nonlocal C, last_C_val, last_C_change_time, crossed_ids, prev_positions, trails, lost_tracks, stitch_count, click_times, history_data
        # 记录历史（如果C不为0）
        if C != 0:
            history_data.append(C)
            # 只保留最近120个记录（8列x15行）
            if len(history_data) > 200:
                history_data = history_data[-200:]
            # 更新UI历史显示
            if ui:
                ui.update_history(history_data)

        C = 0
        crossed_ids.clear()
        prev_positions.clear()
        trails.clear()
        lost_tracks.clear()
        stitch_count = 0
        last_C_val = 0
        last_C_change_time = time.time()
        click_times.clear()

    # ---- 鼠标右键三连击检测 ----
    click_times = []

    def _mouse_callback(event, x, y, flags, param):
        nonlocal click_times
        if event == cv2.EVENT_RBUTTONDOWN:
            now = time.time()
            click_times[:] = [t for t in click_times if now - t < 0.5]
            click_times.append(now)
            if len(click_times) >= 3:
                _reset_counter()
                logger.info("[清零] 鼠标右键三连击")
                click_times.clear()

    # 在无UI模式下设置鼠标回调
    if args.no_ui:
        cv2.namedWindow("猪只计数监控")
        cv2.setMouseCallback("猪只计数监控", _mouse_callback)

    crossed_ids = set()
    prev_positions = {}
    trails = {}
    unique_track_ids = set()
    pig_count = 0
    tracks = []
    inference_frame = None  # 推理结果对应的帧（用于HUD绘制，防止坐标错位）

    # ---- ID缝合参数 ----
    STITCH_ANGLE_THRESH = math.radians(30)   # 方向夹角阈值(±30°)
    STITCH_TIME_MAX = 3                    # 丢失最长时间(秒)
    STITCH_DIST_MAX = 300                    # 预估位置最大距离(px)
    STITCH_SPEED_RATIO_MAX = 3.0            # 速度变化比上限
    STITCH_OCCUPY_DIST = 80                  # 占据检查距离(px)
    lost_tracks = {}                         # 丢失轨迹 {id: {cx, cy, vx, vy, angle, crossed, lost_time}}
    stitch_count = 0                         # 缝合成功次数

    # hou模式录制器（主线程录制含HUD帧）
    hou_recorder = VideoRecorder(cfg.save_dir, cfg.segment_duration, fps, width, height) if cfg.record_mode == "hou" else None

    stat_infer_count = 0
    sum_preprocess_ms = 0.0
    sum_infer_only_ms = 0.0
    sum_postprocess_ms = 0.0
    infer_fps = 0.0

    last_stats_time = time.time()

    def signal_handler(sig, frame_sig):
        nonlocal global_quit
        global_quit = True

    signal.signal(signal.SIGINT, signal_handler)
    print("[系统] 运行中... (Q键/C键/Ctrl+C 退出)")

    # 初始化UI（如果启用）
    ui = None
    ui_root = None
    if not args.no_ui:
        def on_start():
            nonlocal global_quit
            global_quit = False
            logger.info("[UI] 启动信号")
            _ui_ctl.status_msg = "正在连接视频源..."
            start_system()

        def on_stop():
            logger.info("[UI] 停止信号")
            _reset_counter()  # 停止时记录当前C值到历史并归零
            stop_system()

        def on_reset():
            _reset_counter()
            logger.info("[UI] 清零信号")
            _ui_ctl.status_msg = "C值已归零"

        # 创建UI根窗口
        ui_root = tk.Tk()
        ui_root.title("猪只计数监控系统")
        ui_root.state('zoomed')  # 默认最大化
        ui_root.protocol("WM_DELETE_WINDOW", lambda: on_stop())

        # 创建UI实例
        ui = CountingUI(cfg, on_start, on_stop, on_reset, ui_root)
        logger.info("[UI] 界面已初始化")

        # UI模式下绑定快捷键和鼠标右键清零（不受窗口焦点影响）
        ui_root.bind('<KeyPress-c>', lambda e: on_reset())
        ui_root.bind('<KeyPress-C>', lambda e: on_reset())
        _ui_click_times = []
        def _ui_right_click(event):
            nonlocal _ui_click_times
            now = time.time()
            _ui_click_times[:] = [t for t in _ui_click_times if now - t < 0.5]
            _ui_click_times.append(now)
            if len(_ui_click_times) >= 3:
                on_reset()
                logger.info("[清零] 鼠标右键三连击")
                _ui_click_times.clear()
        ui_root.bind('<Button-3>', _ui_right_click)

    # ====== 主线程显示循环 ======
    drop_rate = 0.0  # 初始化丢帧率
    while True:
        # 处理UI事件（如果启用UI）
        if ui:
            try:
                ui.root.update()
            except:
                pass
            # 检查UI是否关闭
            if ui.window_closed:
                global_quit = True
                break

            # 始终更新UI底部信息（状态消息、总数、时间、丢帧计数）
            ui.total_label.config(text=f"总数: {sum(history_data) + C}")
            if _ui_ctl.status_msg:
                ui.status_msg_var.set(_ui_ctl.status_msg)
                _ui_ctl.status_msg = ""
            ui.time_label.config(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            with stats_lock:
                cd = cap_drop_count
                rd = result_drop_count
            ui.drop_label.config(text=f"采丢:{cd} 结丢:{rd}")

            # 如果系统未运行，等待用户点击开始
            if not system_running:
                time.sleep(0.1)
                continue
        else:
            # 无UI模式，自动启动系统
            if not system_running:
                start_system()
                system_running = True

        try:
            latest_results, latest_fid, latest_t_submit = result_queue.get(timeout=0.05)
            with frame_buffer_lock:
                inference_frame = frame_buffer.pop(latest_fid, None)
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
                        prev_side = info['cy'] - line_y_at(info['cx'])
                        cur_side = nt['cy'] - line_y_at(nt['cx'])
                        if prev_side < 0 and cur_side >= 0:
                            C += 1
                            crossed_ids.add(best_old_id)
                            prev_positions[best_old_id] = (nt['cx'], nt['cy'])
                            logger.info(f"[缝合计数] #{best_old_id} 上→下 (+1) C:{C}")
                        elif prev_side > 0 and cur_side <= 0:
                            C -= 1
                            crossed_ids.add(best_old_id)
                            prev_positions[best_old_id] = (nt['cx'], nt['cy'])
                            logger.info(f"[缝合计数] #{best_old_id} 下→上 (-1) C:{C}")
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
                            prev_side = info['cy'] - line_y_at(info['cx'])
                            cur_side = cur['cy'] - line_y_at(cur['cx'])
                            if prev_side < 0 and cur_side >= 0:
                                C += 1
                                crossed_ids.add(tid)
                                prev_positions[tid] = (cur['cx'], cur['cy'])
                                logger.info(f"[恢复计数] #{tid} 上→下 (+1) C:{C}")
                            elif prev_side > 0 and cur_side <= 0:
                                C -= 1
                                crossed_ids.add(tid)
                                prev_positions[tid] = (cur['cx'], cur['cy'])
                                logger.info(f"[恢复计数] #{tid} 下→上 (-1) C:{C}")

            # ---- 越线计数 ----
            for t in tracks:
                tid = t['id']
                cx, cy = t['cx'], t['cy']

                if tid in prev_positions:
                    prev_cx, prev_cy = prev_positions[tid]
                    prev_side = prev_cy - line_y_at(prev_cx)
                    cur_side = cy - line_y_at(cx)
                    if prev_side < 0 and cur_side >= 0:
                        C += 1
                        logger.info(f"[计数] #{tid} 上→下 (+1) C:{C}")
                    elif prev_side > 0 and cur_side <= 0:
                        C -= 1
                        logger.info(f"[计数] #{tid} 下→上 (-1) C:{C}")

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
                    f.write(f"时间：{now_str}, 帧数：{display_count}, C: {C}\n")
                _reset_counter()
                logger.info("[清零] C 30秒不变，已归零")
                _ui_ctl.status_msg = "C值已归零"


        # HUD绘制（使用推理结果对应的帧，而非latest_frame，避免坐标错位）
        if inference_frame is not None:
            frame = inference_frame
        else:
            with frame_lock:
                frame = latest_frame

        display_frame = None
        if frame is not None and frame.size > 0:
            p1 = (int(width * cfg.poly_p1[0] / 100.0), int(height * cfg.poly_p1[1] / 100.0))
            p2 = (int(width * cfg.poly_p2[0] / 100.0), int(height * cfg.poly_p2[1] / 100.0))
            p3 = (int(width * cfg.poly_p3[0] / 100.0), int(height * cfg.poly_p3[1] / 100.0))
            cv2.line(frame, p1, p2, (0, 255, 0), 2)
            cv2.line(frame, p2, p3, (0, 255, 0), 2)

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

            # OpenCV显示（如果未启用UI）
            if args.no_ui:
                cv2.imshow("猪只计数监控", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:
                    global_quit = True
                elif key == ord('c'):
                    _reset_counter()
                    logger.info("[清零] 键盘c键")

        # 更新视频帧和实时状态标签（C值/FPS/丢帧率）
        if ui:
            ui.update_frame(display_frame, C, pig_count, infer_fps, drop_rate)

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
                _ui_ctl.status_msg = (f"{elapsed:.0f}s | 采:{cc} 推:{ic} 显:{dc} | "
                                      f"丢帧:{total_drop}({drop_rate:.1f}%) 采丢:{cd} 结丢:{rd}")
            last_stats_time = now_t

    # ====== 清理 ======
    global_quit = True

    # 关闭UI
    if ui and ui.root:
        try:
            ui.root.quit()
            ui.root.destroy()
        except:
            pass

    cv2.destroyAllWindows()
    for _ in range(30):
        cv2.waitKey(1)

    try:
        infer_queue.put_nowait(None)
    except Full:
        pass

    if t_infer:
        t_infer.join(timeout=3)
    if t_capture:
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
