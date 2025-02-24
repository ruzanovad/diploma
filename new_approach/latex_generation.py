import subprocess
import os
import glob
import random
from PIL import Image
from dotenv import load_dotenv
import utils


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


def fill_file(images_dir, labels_dir, code):

    choice = random.randint(0, 2)
    if choice == 0:
        content = generate_number()
        suffix = "number"
    elif choice == 1:
        content = generate_decimal()
        suffix = "decimal"
    else:
        content = generate_word()
        suffix = "word"

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


def generate_word(length=5) -> str:
    length = random.randint(1, 5)
    return "".join(
        random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", k=length)
    )


def generate_dataset(level="number", count=1000, seed=42, train=80, val=20):
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
    for code in range(train_number):
        fill_file(images_train_dir, labels_train_dir, code)

    for code in range(train_number, train_number + val_number):
        fill_file(images_val_dir, labels_val_dir, code)

    # Clean up
    for ext in ["aux", "log", "dvi", "tex"]:
        delete_files_with_extension(images_train_dir, ext)
        delete_files_with_extension(images_val_dir, ext)

    utils.generate_yolo_yaml(
        os.getenv("templates"), os.getenv("dataset_folder"), "dataset.yaml"
    )


if __name__ == "__main__":
    # generate_pattern()
    generate_dataset()
    # print(load_symbols_from_templates(os.getenv("templates")))
