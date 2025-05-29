import os
import numpy as np
import yaml
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm


SUPPORTS_LIMITS = {"\\int", "\\sum", "\\prod", "\\lim"}


def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def get_width(bbox):
    x1, _, x2, _ = bbox
    return x2 - x1


def get_height(bbox):
    _, y1, _, y2 = bbox
    return y2 - y1


def horizontally_overlap(b1, b2, thr=0.6):
    """True if b1 and b2 overlap by ≥ thr·min(widths)."""
    overlap = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    return overlap > thr * min(get_width(b1), get_width(b2))


def find_radicand(idx, boxes, y_tol=0.3):
    """For a √ at boxes[idx], find the nearest box to its right on roughly the same y."""
    xi, yi = get_center(boxes[idx]["bbox"])
    hi = get_height(boxes[idx]["bbox"])
    best_j, best_dx = None, float("inf")

    for j, b in enumerate(boxes):
        if j == idx: 
            continue
        xj, yj = get_center(b["bbox"])
        dy = abs(yj - yi)
        dx = xj - xi
        if dx > 0 and dy <= y_tol * hi and dx < best_dx:
            best_dx, best_j = dx, j

    return best_j


def build_relation_graph(boxes, x_gap=0.6, y_gap=0.35):
    """
    Adds to each dict in `boxes` a key 'relations': a list of {'type':..., 'child':j}.
    """
    # 1) clear/init
    for b in boxes:
        b["relations"] = []

    # 2) sort by x center
    order = sorted(range(len(boxes)),
                   key=lambda i: get_center(boxes[i]["bbox"])[0])

    # 3) detect super/sub (or over/under) relations
    for i in order:
        parent = boxes[i]
        xi, yi = get_center(parent["bbox"])
        hi = get_height(parent["bbox"])
        is_limit = parent["label"] in SUPPORTS_LIMITS

        for j, child in enumerate(boxes):
            if i == j:
                continue
            if not horizontally_overlap(parent["bbox"], child["bbox"], thr=x_gap):
                continue

            _, yj = get_center(child["bbox"])
            dy = (yi - yj) / hi

            if dy > y_gap:
                rel = "under" if is_limit else "sup"
            elif dy < -y_gap:
                rel = "over" if is_limit else "sub"
            else:
                continue

            parent["relations"].append({"type": rel, "child": j})

    # 4) √ → radicand
    for i, b in enumerate(boxes):
        if b["label"] == "\\sqrt":
            j = find_radicand(i, boxes)
            if j is not None:
                b["relations"].append({"type": "radicand", "child": j})


def to_latex(boxes):
    visited = set()

    def dfs(i):
        if i in visited:
            return ""
        visited.add(i)

        node = boxes[i]["label"]
        rels = boxes[i]["relations"]

        # radicand first
        rad = [r["child"] for r in rels if r["type"] == "radicand"]
        if rad:
            return f"\\sqrt{{{dfs(rad[0])}}}"

        # fraction if over+under
        over = [r["child"] for r in rels if r["type"] == "over"]
        under = [r["child"] for r in rels if r["type"] == "under"]
        if over and under:
            return f"\\frac{{{dfs(over[0])}}}{{{dfs(under[0])}}}"

        # limits vs normal sup/sub
        is_limit = node in SUPPORTS_LIMITS
        if is_limit and over and under:
            node = f"{node}\\limits"

        sup = [r["child"] for r in rels if r["type"] == "sup"]
        sub = [r["child"] for r in rels if r["type"] == "sub"]

        out = node
        if sup:
            out += f"^{{{''.join(dfs(c) for c in sup)}}}"
        if sub:
            out += f"_{{{''.join(dfs(c) for c in sub)}}}"
        return out

    parts = []
    for i in sorted(range(len(boxes)),
                    key=lambda k: get_center(boxes[k]["bbox"])[0]):
        if i not in visited:
            parts.append(dfs(i))

    return "".join(parts)


def denorm(box, img_w, img_h):
    """
    YOLO → absolute [x1,y1,x2,y2].
    `box = (xc, yc, bw, bh)` all normalized [0..1].
    """
    xc, yc, bw, bh = box
    x1 = (xc - bw / 2) * img_w
    y1 = (yc - bh / 2) * img_h
    x2 = (xc + bw / 2) * img_w
    y2 = (yc + bh / 2) * img_h
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
        img_h, img_w = img.shape[:2]

        try:
            txt = np.loadtxt(pred_path).reshape(-1, 5)
        except Exception as e:
            print(f"Error loading {pred_path}: {e}")
            continue

        boxes = []
        for row in txt:
            class_id, xc, yc, bw, bh = row
            class_id = int(class_id)
            bbox = denorm([xc, yc, bw, bh], img_w, img_h)

            boxes.append(
                {
                    "filename": filename,
                    "class_id": class_id,
                    "label": class_map[class_id],
                    "bbox": bbox,
                }
            )

        boxes = sorted(boxes, key=lambda x: x["bbox"][0])
        build_relation_graph(boxes)  # <--- Add this line
        texts.append(to_latex(boxes))
        all_boxes.append(boxes)

    if all_boxes:
        print(all_boxes[1])
        print(test_formulas[1])
        print(texts[1])
    else:
        print("No predictions found.")
