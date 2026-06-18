#!/usr/bin/env python3
import os
import random
import cv2
from pathlib import Path

src_dir = r"E:\pig\src"
save_dir = r"E:\pig\extracted_frames"
os.makedirs(save_dir, exist_ok=True)

exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
videos = sorted([f for f in Path(src_dir).iterdir() if f.suffix.lower() in exts])
print(f"找到 {len(videos)} 个视频")

idx = 0
for video_path in videos:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  跳过 {video_path.name}")
        continue
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps > 0 else 0
    print(f"  {video_path.name}: {duration:.1f}s {fps:.0f}fps")

    t = 0.0
    count = 0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        cv2.imwrite(os.path.join(save_dir, f"{idx:04d}.jpg"), frame)
        idx += 1
        count += 1
        t += random.uniform(0.3, 0.8)

    cap.release()
    print(f"    抽取 {count} 帧")

print(f"\n完成, 保存至 {save_dir}, 共 {idx} 张")
