import subprocess
import os
import glob
import random
from PIL import Image
import bounding_box


latex = r"""\documentclass{standalone}
\usepackage{amsmath}
\begin{document}
%s
\end{document}
"""

DATASETS = "datasets"

def delete_files_with_extension(directory, extension):
    for file_path in glob.glob(os.path.join(directory, f"*.{extension}")):
        try:
            os.remove(file_path)
            # print(f"Deleted: {file_path}")
        except Exception as e:
            pass
            # print(f"Error deleting {file_path}: {e}")


def generate_pattern(level="number"):
    match level:
        case "number":
            base_dir = os.path.join(DATASETS , "number_patterns")
            os.makedirs(base_dir, exist_ok=True)

            for number in map(str, range(0, 10)):
                current_dir = os.path.join(base_dir, number)
                os.makedirs(current_dir, exist_ok=True)

                for code in range(1):
                    current_file = os.path.join(current_dir, str(code))

                    tex_file = f"{current_file}.tex"
                    txt_file = f"{current_file}.txt"
                    with open(tex_file, "w") as f:
                        f.write(latex % ("$" + number + "$"))

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
                        print(
                            f"Error compiling {tex_file}: {tex_result.stderr.decode()}"
                        )
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
                        file.write(f"{number} {width/2} {height/2} {width} {height}\n")

                    # Clean up
                    for ext in ["aux", "log", "dvi", "tex"]:
                        delete_files_with_extension(current_dir, ext)

        case _:
            raise Exception("Unsupported pattern level")

def fill_file(images_dir, labels_dir, code, level, unique_numbers):
    current_file = os.path.join(images_dir, "%d_%s" % (code, level))

    tex_file = f"{current_file}.tex"
    with open(tex_file, "w") as f:
        f.write(latex % unique_numbers[code])

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
        print(
            f"Error converting {dvi_file} to PNG: {dvi_result.stderr.decode()}"
        )
        return

    txt_file = os.path.join(labels_dir, "%d_%s" % (code, level))+'.txt'
    bounding_boxes = bounding_box.get_bounding_boxes(png_file)

    with open(txt_file, "w") as file:
        for box in bounding_boxes:
            file.write(' '.join(box) +'\n')
    



def generate_dataset(level="number", count=100, seed=42, train=80, val=20):
    assert train + val == 100
    random.seed(seed)
    match level:
        case "number":
            base_dir = os.path.join(DATASETS , "dataset")
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

            unique_numbers = random.sample(range(1, 10001), count)


            train_number = (count * train) // 100
            val_number = count - train_number
            for code in range(train_number):
                fill_file(images_train_dir, labels_train_dir, code, level,unique_numbers)
            
            for code in range(train_number, train_number + val_number):
                fill_file(images_val_dir, labels_val_dir, code, level,unique_numbers)
                
            # Clean up
            for ext in ["aux", "log", "dvi", "tex"]:
                delete_files_with_extension(images_train_dir, ext)
                delete_files_with_extension(images_val_dir, ext)
            
            with open("dataset.yaml", "w") as file:
                file.write("path: dataset\n")
                file.write("train: images/train\n")
                file.write("val: images/val\n\n")
                file.write("names:\n")
                for key in sorted(bounding_box.types.keys()):
                    file.write(f"  {key}: {bounding_box.types[key]}\n")

        case _:
            raise Exception("Unsupported pattern level")


if __name__ == "__main__":
# generate_pattern()
    generate_dataset()
