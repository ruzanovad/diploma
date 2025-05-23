import os
import numpy as np
import yaml
import cv2
import pandas as pd


with open("experiment/dataset.yaml", "r") as f:
    class_map = yaml.safe_load(f)["names"]

path_pred = "notebooks/runs/detect/val/labels"
path_images = "datasets/experiment/dataset/images/test"  # denormalize coordinates
path_csv = "datasets/experiment/experiment.csv"
pred_labels = []


def denorm(box, w, h):
    xc, yc, bw, bh = box
    x1 = (xc - bw / 2) * w
    y1 = (yc - bh / 2) * h
    x2 = (xc + bw / 2) * w
    y2 = (yc + bh / 2) * h
    return [x1, y1, x2, y2]


all_boxes = []

# Получаем список всех файлов с предсказаниями
df = pd.read_csv("datasets/experiment/experiment.csv")
test_formulas = df[df['split'] == 'test']['formula'].tolist()
label_files = sorted(os.path.join(path_pred, "*.txt"))

for pred_path in label_files:
    filename = os.path.splitext(os.path.basename(pred_path))[0]
    image_path = os.path.join(path_images, f"{filename}.png")
    if not os.path.exists(image_path):
        print(f"Image not found for {filename}")
        continue


    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image {image_path}")
        continue
    h, w = img.shape[:2]

    try:
        txt = np.loadtxt(pred_path).reshape(-1, 5)  # (N, 5)
    except Exception as e:
        print(f"Error loading {pred_path}: {e}")
        continue

    # Обработка каждой строки
    boxes = []
    for row in txt:
        class_id, xc, yc, bw, bh = row
        class_id = int(class_id)
        bbox = denorm([xc, yc, bw, bh], w, h)

        boxes.append(
            {
                "filename": filename,
                "class_id": class_id,
                "label": class_map[class_id],
                "bbox": bbox,  # [x1, y1, x2, y2]
            }
        )

    all_boxes.append(boxes)


if all_boxes:
    print(all_boxes[0]) 
else:
    print("No predictions found.")
