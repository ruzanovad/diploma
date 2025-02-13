import cv2
import os
import numpy as np
import latex_generation


DATASETS = "datasets"
patterns_dir = os.path.join(DATASETS, "number_patterns")

types = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}


def get_bounding_boxes(filename, threshold=0.9):
    """
    Get bounding box in YOLO format
    """
    boxes = []
    for pattern in types.keys():
        # Загрузим изображение и шаблон
        image = cv2.imread(filename, 0)
        template = cv2.imread(os.path.join(patterns_dir, types[pattern], "0.png"), 0)

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
