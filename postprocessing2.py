import os
import numpy as np
import yaml
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm

SUPPORTS_LIMITS = {"\\int", "\\sum", "\\prod", "\\lim"}

def ctr(b):
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2, (y1 + y2) / 2

def w(b):
    x1, _, x2, _ = b
    return x2 - x1

def h(b):
    _, y1, _, y2 = b
    return y2 - y1

def horizontally_close(b1, b2, thr=0.6):
    x11, _, x12, _ = b1
    x21, _, x22, _ = b2
    overlap = max(0, min(x12, x22) - max(x11, x21))
    return overlap > thr * min(w(b1), w(b2))

def find_radicand(i, boxes):
    xi, yi = ctr(boxes[i]["bbox"])
    hi = h(boxes[i]["bbox"])
    cand = None
    dx_min = float("inf")
    for j, b in enumerate(boxes):
        if j == i:
            continue
        xj, yj = ctr(b["bbox"])
        if yi - 0.3 * hi < yj < yi + 0.3 * hi and xj > xi:
            dx = xj - xi
            if dx < dx_min:
                dx_min = dx
                cand = j
    return cand

def build_relation_graph(boxes, x_gap=0.6, y_gap=0.35):
    """
    Для каждого элемента boxes добавляется ключ "relations" со списком
    {"type": ..., "child": j}.
    """
    n = len(boxes)
    # очищаем или инициализируем
    for b in boxes:
        b["relations"] = []

    order = sorted(range(n), key=lambda i: ctr(boxes[i]["bbox"])[0])

    for i in order:
        bi = boxes[i]["bbox"]
        xi, yi = ctr(bi)
        hi = h(bi)
        is_limit = boxes[i]["label"] in SUPPORTS_LIMITS

        for j in range(n):
            if i == j:
                continue
            bj = boxes[j]["bbox"]
            if not horizontally_close(bi, bj, thr=x_gap):
                continue
            xj, yj = ctr(bj)
            dy = (yi - yj) / hi

            if dy > y_gap:
                rel = "sup"
            elif dy < -y_gap:
                rel = "sub"
            else:
                continue
            boxes[i]["relations"].append({"type": rel, "child": j})

    for i, b in enumerate(boxes):
        if b["label"] in {"√", "sqrt"}:
            cand = find_radicand(i, boxes)
            if cand is not None:
                boxes[i]["relations"].append({"type": "radicand", "child": cand})

def to_latex(boxes):
    visited = set()

    def dfs(i):
        if i in visited:
            return ""
        visited.add(i)
        node = boxes[i]["label"]

        # собираем связи
        rels = boxes[i]["relations"]
        sup = [e["child"] for e in rels if e["type"] == "sup"]
        sub = [e["child"] for e in rels if e["type"] == "sub"]
        over = [e["child"] for e in rels if e["type"] == "over"]
        under = [e["child"] for e in rels if e["type"] == "under"]
        rad = [e["child"] for e in rels if e["type"] == "radicand"]

        result = node
        if rad:
            result = f"\\sqrt{{{dfs(rad[0])}}}"
        if over and under:
            result = f"\\frac{{{dfs(over[0])}}}{{{dfs(under[0])}}}"
        if sup:
            result += f"^{{{''.join(dfs(k) for k in sup)}}}"
        if sub:
            result += f"_{{{''.join(dfs(k) for k in sub)}}}"
        return result

    latex = []
    for i in sorted(range(len(boxes)), key=lambda k: ctr(boxes[k]["bbox"])[0]):
        if i not in visited:
            latex.append(dfs(i))
    return "".join(latex)



def denorm(box, w, h):
    xc, yc, bw, bh = box
    x1 = (xc - bw / 2) * w
    y1 = (yc - bh / 2) * h
    x2 = (xc + bw / 2) * w
    y2 = (yc + bh / 2) * h
    return [x1, y1, x2, y2]


with open("datasets/experiment/dataset.yaml", "r") as f:
    class_map = yaml.safe_load(f)["names"]


if __name__ == "__main__":
    path_pred = "notebooks/runs/detect/val/labels"
    path_images = "datasets/experiment/dataset/images/test"  # denormalize coordinates
    path_csv = "datasets/experiment/experiment.csv"
    pred_labels = []

    texts = []
    all_boxes = []

    # Получаем список всех файлов с предсказаниями
    df = pd.read_csv("datasets/experiment/experiment.csv")
    test_formulas = df[df["split"] == "test"]["formula"].tolist()
    label_files = sorted(glob(os.path.join(path_pred, "*.txt")))

    for pred_path in tqdm(label_files):
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

        boxes = sorted(boxes, key=lambda x: x["bbox"][0])

        text = [x["label"] for x in boxes]
        texts.append(" ".join(text))

        all_boxes.append(boxes)

    if all_boxes:
        print(all_boxes[1])
        print(test_formulas[1])
        print(texts[1])
    else:
        print("No predictions found.")
