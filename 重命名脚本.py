#!/usr/bin/env python3
import os
import random
import shutil
from pathlib import Path

src_dir = r"E:\pig\extracted_frames"
dst_dir = r"E:\pig\extracted_frames_renamed"
os.makedirs(dst_dir, exist_ok=True)

images = sorted(Path(src_dir).glob("*.jpg")) + sorted(Path(src_dir).glob("*.png"))
print(f"共 {len(images)} 张图片")

used = set()
for img in images:
    while True:
        name = f"{random.randint(0, 9999):04d}"
        if name not in used:
            break
    used.add(name)
    shutil.copy2(str(img), os.path.join(dst_dir, f"{name}{img.suffix}"))

print(f"完成, 保存至 {dst_dir}, 共 {len(used)} 张")
