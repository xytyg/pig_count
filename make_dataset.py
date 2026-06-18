#!/usr/bin/env python3
import os
import random
import shutil
from pathlib import Path
from ultralytics import YOLO

src_dir = r"E:\pig\extracted_frames_renamed"
dst_dir = r"E:\pig\dataset"
model_path = r"E:\pig\src\best8_1280.pt"
conf = 0.7
iou = 0.8
train_ratio = 0.8

train_img_dir = os.path.join(dst_dir, "train", "images")
train_lbl_dir = os.path.join(dst_dir, "train", "labels")
val_img_dir = os.path.join(dst_dir, "val", "images")
val_lbl_dir = os.path.join(dst_dir, "val", "labels")
for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
    os.makedirs(d, exist_ok=True)

images = sorted(Path(src_dir).glob("*.jpg")) + sorted(Path(src_dir).glob("*.png"))
random.shuffle(images)
split_idx = int(len(images) * train_ratio)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]
print(f"共 {len(images)} 张, 训练集 {len(train_imgs)}, 验证集 {len(val_imgs)}")

for img, is_train in [(p, True) for p in train_imgs] + [(p, False) for p in val_imgs]:
    img_dir = train_img_dir if is_train else val_img_dir
    shutil.copy2(str(img), os.path.join(img_dir, img.name))

print("图片划分完成, 开始推理标注...")

model = YOLO(model_path)
class_names = model.names

def predict_and_save(img_dir, lbl_dir, tag):
    imgs = sorted(Path(img_dir).glob("*.jpg")) + sorted(Path(img_dir).glob("*.png"))
    total_boxes = 0
    for i, img_path in enumerate(imgs):
        results = model.predict(source=str(img_path), conf=conf, iou=iou, verbose=False)
        r = results[0]
        h, w = r.orig_shape[:2]
        lines = []
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                total_boxes += 1
        lbl_path = os.path.join(lbl_dir, img_path.stem + ".txt")
        with open(lbl_path, "w") as f:
            f.write("\n".join(lines))
        if (i + 1) % 100 == 0 or i == len(imgs) - 1:
            print(f"  [{tag}] {i+1}/{len(imgs)} 累计标注:{total_boxes}")
    return total_boxes

t1 = predict_and_save(train_img_dir, train_lbl_dir, "train")
t2 = predict_and_save(val_img_dir, val_lbl_dir, "val")

data_yaml = os.path.join(dst_dir, "data.yaml")
with open(data_yaml, "w", encoding="utf-8") as f:
    f.write(f"path: {dst_dir}\n")
    f.write(f"train: train/images\n")
    f.write(f"val: val/images\n")
    f.write(f"names:\n")
    for k, v in sorted(class_names.items()):
        f.write(f"  {k}: {v}\n")

print(f"\n完成!")
print(f"  训练集: {len(train_imgs)}张 {t1}个标注")
print(f"  验证集: {len(val_imgs)}张 {t2}个标注")
print(f"  data.yaml: {data_yaml}")