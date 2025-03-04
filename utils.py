import os
import yaml
from dotenv import load_dotenv
import cv2
import numpy as np
import torch


def get_bounding_boxes(filename, class_dict:dict, threshold=0.9):
    """
    Get bounding box in YOLO format
    """
    load_dotenv()
    patterns_dir = os.getenv("patterns_folder")

    # types = load_symbols_from_templates(os.getenv("templates"))

    boxes = []
    for pattern in class_dict.keys():
        # Загрузим изображение и шаблон
        image = cv2.imread(filename, 0)
        template = cv2.imread(
            os.path.join(patterns_dir, str(class_dict[pattern]), "0.png"), 0
        )

        w, h = template.shape[::-1]
        img_w, img_h = image.shape[::-1]

        if img_w < w or img_h < h:
            continue

        # Выполним сопоставление
        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

        # Установим порог обнаружения
        # threshold = 0.8
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            boxes.append(
                list(
                    map(
                        str,
                        [
                            class_dict[pattern],
                            str((pt[0] + w * 0.5) / img_w),
                            str((pt[1] + h * 0.5) / img_h),
                            str(w / img_w),
                            str(h / img_h),
                        ],
                    )
                )
            )
    return boxes


def load_symbols_from_templates(template_dir):
    symbols_dict = {}
    code = 0

    for template_file in os.listdir(template_dir):
        if template_file.endswith(".txt"):
            with open(os.path.join(template_dir, template_file), "r") as file:
                for line in file:
                    symbol = line.strip()
                    if symbol not in symbols_dict:
                        symbols_dict[symbol] = code
                        code += 1

    return symbols_dict


def generate_yolo_yaml(template_dir, dataset_dir, output_file):
    symbols_dict = load_symbols_from_templates(template_dir)

    classes = {symbols_dict[key]: key for key in symbols_dict.keys()}

    yolo_config = {
        "path": dataset_dir,
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": classes,
    }

    with open(output_file, "w") as file:
        yaml.dump(yolo_config, file, default_flow_style=False)


def get_inverted_dict(dictionary):
    return {v: k for k, v in dictionary.items()}


def stupid_encoder(bounding_boxes: torch.Tensor, class_to_latex: dict = None):
    """
    Encodes bounding boxes into LaTeX code using naive approach.

    Parameters:
    bounding_boxes (torch.Tensor): Tensor of bounding boxes with format [class, x_center, y_center, width, height].
    class_to_latex (dict): Dictionary mapping classes to LaTeX tags.

    Returns:
    str: Generated LaTeX code.
    """
    if class_to_latex == None:
        class_to_latex = get_inverted_dict(
            load_symbols_from_templates(os.getenv("templates"))
        )

    bounding_boxes = bounding_boxes.to("cpu")
    # Сортировка по x, потом по y

    def lexsort(keys, dim=-1):
        if keys.ndim < 2:
            raise ValueError(f"keys must be at least 2 dimensional, but {keys.ndim=}.")
        if len(keys) == 0:
            raise ValueError(f"Must have at least 1 key, but {len(keys)=}.")

        idx = keys[0].argsort(dim=dim, stable=True)
        for k in keys[1:]:
            idx = idx.gather(dim, k.gather(dim, idx).argsort(dim=dim, stable=True))

        return idx

    sorted_indices = lexsort((bounding_boxes[:, 1], bounding_boxes[:, 2]))
    sorted_boxes = bounding_boxes[sorted_indices]
    # Получаем class_id как long-тензор (для индексирования в словаре)
    class_ids = sorted_boxes[:, 0].long()

    # Генерация LaTeX-кода (собираем строки через join для эффективности)
    latex_code = "".join(
        class_to_latex.get(int(class_id), "") for class_id in class_ids
    )

    return latex_code
