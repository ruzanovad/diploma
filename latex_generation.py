import subprocess
import os
import glob
import random
from PIL import Image
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor # for concurrency
from functools import partial # for concurrency
import utils
import yaml
import numpy as np


load_dotenv()

latex = r"""\documentclass{standalone}
\usepackage{amsmath}
\begin{document}
%s
\end{document}
"""


def delete_files_with_extension(directory, extension):
    for file_path in glob.glob(os.path.join(directory, f"*.{extension}")):
        try:
            os.remove(file_path)
            # print(f"Deleted: {file_path}")
        except Exception as e:
            pass
            # print(f"Error deleting {file_path}: {e}")


def generate_pattern():
    base_dir = os.getenv("patterns_folder")

    os.makedirs(base_dir, exist_ok=True)

    templates = utils.load_symbols_from_templates(os.getenv("templates"))

    for token, class_number in templates.items():
        # for number in map(str, range(0, 10)):
        current_dir = os.path.join(base_dir, str(class_number))
        os.makedirs(current_dir, exist_ok=True)

        for code in range(1):
            current_file = os.path.join(current_dir, str(code))

            tex_file = f"{current_file}.tex"
            txt_file = f"{current_file}.txt"
            with open(tex_file, "w") as f:
                f.write(latex % ("$" + token + "$"))

            # Compile .tex to .dvi
            tex_result = subprocess.run(
                [
                    "latex",
                    "-interaction=nonstopmode",
                    os.path.basename(tex_file),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=current_dir,
            )
            if tex_result.returncode != 0:
                print(f"Error compiling {tex_file}: {tex_result.stderr.decode()}")
                continue

            # Convert .dvi to .png
            dvi_file = f"{current_file}.dvi"
            png_file = f"{current_file}.png"
            dvi_result = subprocess.run(
                [
                    "dvipng",
                    "-T",
                    "tight",
                    "-D",
                    "300",
                    "-o",
                    os.path.basename(png_file),
                    os.path.basename(dvi_file),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=current_dir,
            )
            if dvi_result.returncode != 0:
                print(
                    f"Error converting {dvi_file} to PNG: {dvi_result.stderr.decode()}"
                )
                continue

            im = Image.open(png_file)
            width, height = im.size

            with open(txt_file, "w") as file:
                file.write(f"{class_number} {width/2} {height/2} {width} {height}\n")

            # Clean up
            for ext in ["aux", "log", "dvi", "tex"]:
                delete_files_with_extension(current_dir, ext)


def generate_one(current_prefix: str, template):
    """generates one pic with template in cwd"""
    current_file = current_prefix
    tex_file = f"{current_file}.tex"
    with open(tex_file, "w") as f:
        f.write(latex % ("$" + template + "$"))

    # Compile .tex to .dvi
    tex_result = subprocess.run(
        [
            "latex",
            "-interaction=nonstopmode",
            os.path.basename(tex_file),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # cwd=images_dir,
    )
    if tex_result.returncode != 0:
        print(f"Error compiling {tex_file}: {tex_result.stderr.decode()}")
        return

    # Convert .dvi to .png
    dvi_file = f"{current_file}.dvi"
    png_file = f"{current_file}.png"
    dvi_result = subprocess.run(
        [
            "dvipng",
            "-T",
            "tight",
            "-D",
            "300",
            "-o",
            os.path.basename(png_file),
            os.path.basename(dvi_file),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # cwd=images_dir,
    )
    if dvi_result.returncode != 0:
        print(f"Error converting {dvi_file} to PNG: {dvi_result.stderr.decode()}")
        return

    for ext in ["aux", "log", "dvi", "tex"]:
        delete_files_with_extension("", ext)


def fill_file_job(args):
    """
    Multiprocessing-compatible function.
    Receives a tuple (code, content, images_dir, labels_dir, verbose, print_every)
    """
    code, content, class_dict, images_dir, labels_dir, verbose = args

    fill_file(images_dir, labels_dir, code, content, class_dict, suffix="")

    if verbose>1 and code % verbose == 0:
        print(f"[INFO] Processed {code} samples...")

def fill_file(images_dir, labels_dir, code, content, class_dict : dict, suffix):

    temp_name = "%d_%s" % (code, suffix)
    current_file = os.path.join(images_dir, temp_name)
    tex_file = f"{current_file}.tex"
    with open(tex_file, "w") as f:
        f.write(latex % ("$" + content + "$"))

    # Compile .tex to .dvi
    tex_result = subprocess.run(
        [
            "latex",
            "-interaction=nonstopmode",
            os.path.basename(tex_file),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=images_dir,
    )
    if tex_result.returncode != 0:
        print(f"Error compiling {tex_file}: {tex_result.stderr.decode()}")
        return

    # Convert .dvi to .png
    dvi_file = f"{current_file}.dvi"
    png_file = f"{current_file}.png"
    dvi_result = subprocess.run(
        [
            "dvipng",
            "-T",
            "tight",
            "-D",
            "300",
            "-o",
            os.path.basename(png_file),
            os.path.basename(dvi_file),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=images_dir,
    )
    if dvi_result.returncode != 0:
        print(f"Error converting {dvi_file} to PNG: {dvi_result.stderr.decode()}")
        return

    txt_file = os.path.join(labels_dir, temp_name) + ".txt"

    bounding_boxes = utils.get_bounding_boxes(png_file, class_dict)

    with open(txt_file, "w") as file:
        for box in bounding_boxes:
            file.write(" ".join(box) + "\n")


def generate_one_with_label(images_dir, labels_dir, code, content: str):
    temp_name = "%d" % (code)
    current_file = os.path.join(images_dir, temp_name)
    tex_file = f"{current_file}.tex"
    with open(tex_file, "w") as f:
        f.write(latex % ("$" + content + "$"))

    # Compile .tex to .dvi
    tex_result = subprocess.run(
        [
            "latex",
            "-interaction=nonstopmode",
            os.path.basename(tex_file),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=images_dir,
    )
    if tex_result.returncode != 0:
        print(f"Error compiling {tex_file}: {tex_result.stderr.decode()}")
        return

    # Convert .dvi to .png
    dvi_file = f"{current_file}.dvi"
    png_file = f"{current_file}.png"
    dvi_result = subprocess.run(
        [
            "dvipng",
            "-T",
            "tight",
            "-D",
            "300",
            "-o",
            os.path.basename(png_file),
            os.path.basename(dvi_file),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=images_dir,
    )
    if dvi_result.returncode != 0:
        print(f"Error converting {dvi_file} to PNG: {dvi_result.stderr.decode()}")
        return

    txt_file = os.path.join(labels_dir, temp_name) + ".txt"
    
    bounding_boxes = utils.get_bounding_boxes(png_file)

    with open(txt_file, "w") as file:
        for box in bounding_boxes:
            file.write(" ".join(box) + "\n")


def generate_number() -> str:
    return str(random.randint(0, 9999))


def generate_decimal() -> str:
    integer_part = random.randint(0, 9999)
    decimal_part = random.randint(0, 9999)
    return f"{integer_part}.{decimal_part}"


def generate_word() -> str:
    length = random.randint(10, 45)
    return "".join(
        random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", k=length)
    )


def generate_greek(list_of_letters: list) -> str:
    length = random.randint(10, 45)
    return "".join(random.choices(list_of_letters, k=length))


def generate_dataset_only_templates():
    """
    for each class only one picture
    """

    base_dir = os.getenv("yolo_dataset_folder")
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")

    images_train_dir = os.path.join(images_dir, "train")
    images_val_dir = os.path.join(images_dir, "val")
    labels_train_dir = os.path.join(labels_dir, "train")
    labels_val_dir = os.path.join(labels_dir, "val")

    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)

    template_dir = os.getenv("templates")
    symbols_dict = utils.load_symbols_from_templates(template_dir)
    classes = {symbols_dict[key]: key for key in symbols_dict.keys()}

    for code in range(len(classes)):
        generate_one_with_label(images_train_dir, labels_train_dir, code, classes[code])

    for code in range(len(classes), 2 * len(classes)):
        fill_file(images_val_dir, labels_val_dir, code)

    # Clean up
    for ext in ["aux", "log", "dvi", "tex"]:
        delete_files_with_extension(images_train_dir, ext)
        delete_files_with_extension(images_val_dir, ext)

    dataset_dir = os.path.basename(os.path.normpath(os.getenv("yolo_dataset_folder")))

    yolo_config = {
        "path": dataset_dir,
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": classes,
    }

    with open("dataset.yaml", "w") as file:
        yaml.dump(yolo_config, file, default_flow_style=False)


def generate_dataset(level="number", count=1000, seed=42, train=90, val=10, verbose=100):
    """
    Generates dataset in parallel using multiprocessing with optional verbose logging.
    """
    assert train + val == 100
    random.seed(seed)

    base_dir = os.getenv("yolo_dataset_folder")
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")

    images_train_dir = os.path.join(images_dir, "train")
    images_val_dir = os.path.join(images_dir, "val")
    labels_train_dir = os.path.join(labels_dir, "train")
    labels_val_dir = os.path.join(labels_dir, "val")

    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)

    train_number = (count * train) // 100
    val_number = count - train_number

    # Load classes from templates
    symbols_dict = utils.load_symbols_from_templates(os.getenv("templates"))
    classes = [
        key if len(key) == 1 else key + " "
        for key in symbols_dict.keys()
    ]
    lengths = np.random.randint(20, 41, count).tolist()

    def get_random_content(idx):
        """Генерирует случайное выражение и список использованных классов"""
        selected_classes = random.choices(classes, k=lengths[idx])
        expression = ''.join(selected_classes)
        return expression, {key.strip(): symbols_dict[key.strip()] for key in selected_classes}

    # Prepare argument lists for parallel execution
    train_args = [(i, *get_random_content(i), images_train_dir, labels_train_dir, verbose) 
                  for i in range(train_number)]
    
    val_args = [(i, *get_random_content(i), images_val_dir, labels_val_dir, verbose) 
                for i in range(train_number, train_number + val_number)]

    # --- Generate the TRAIN set in parallel ---
    with ProcessPoolExecutor(max_workers=6) as executor:
        executor.map(fill_file_job, train_args)

    # --- Generate the VAL set in parallel ---
    with ProcessPoolExecutor(max_workers=6) as executor:
        executor.map(fill_file_job, val_args)

    # Finally clean up aux files
    for ext in ["aux", "log", "dvi", "tex"]:
        delete_files_with_extension(images_train_dir, ext)
        delete_files_with_extension(images_val_dir, ext)

    # Generate dataset.yaml
    utils.generate_yolo_yaml(
        os.getenv("templates"),
        os.path.basename(os.path.normpath(os.getenv("yolo_dataset_folder"))),
        "dataset.yaml",
    )


if __name__ == "__main__":
    # generate_pattern()
    # print("dataset done")
    generate_dataset(count=100)
    # generate_dataset_only_templates()
    # print(load_symbols_from_templates(os.getenv("templates")))
