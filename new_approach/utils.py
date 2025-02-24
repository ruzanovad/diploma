import os
import yaml
from dotenv import load_dotenv
import cv2
import numpy as np

def get_bounding_boxes(filename, threshold=0.9):
    """
    Get bounding box in YOLO format
    """
    load_dotenv()
    patterns_dir = os.getenv("patterns_folder")

    types = load_symbols_from_templates(os.getenv("templates"))

    boxes = []
    for pattern in types.keys():
        # Загрузим изображение и шаблон
        image = cv2.imread(filename, 0)
        template = cv2.imread(os.path.join(patterns_dir, str(types[pattern]), "0.png"), 0)

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
                            types[pattern],
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

    classes = {symbols_dict[key] : key for  key in symbols_dict.keys()}

    yolo_config = {
        'path': dataset_dir,
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': classes
    }

    with open(output_file, 'w') as file:
        yaml.dump(yolo_config, file, default_flow_style=False)
