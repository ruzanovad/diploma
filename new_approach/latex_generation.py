import subprocess
import os
import glob
import json
import random
import re


latex = r"""\documentclass{standalone}
\usepackage{amsmath}
\begin{document}
%s
\end{document}
"""


# def parse_dvi_log(dvi_log):
#     symbols = []
#     dvi_to_pixel = 0.00006334  # Conversion factor

#     for line in dvi_log.split("\n"):
#         match = re.search(r"setchar(\d+) h:=(-?\d+)", line)
#         if match:
#             # print("!")
#             char_code = int(match.group(1))  # ASCII code of the symbol
#             x_position = int(match.group(2))  # DVI horizontal position

#             # Convert DVI coordinates to pixels
#             x_pixel = x_position * dvi_to_pixel
#             symbol = chr(char_code)  # Convert ASCII code to character

#             # Approximate bounding box size (based on font size)
#             bbox_width = 10  # Estimated width
#             bbox_height = 14  # Estimated height

#             # Store extracted data
#             symbols.append(
#                 {
#                     "symbol": symbol,
#                     "x_min": int(x_pixel),
#                     "y_min": 0,  # Adjust with real vertical parsing
#                     "x_max": int(x_pixel + bbox_width),
#                     "y_max": 0 + bbox_height,  # Adjust as needed
#                 }
#             )

#     return symbols


def delete_files_with_extension(directory, extension):
    for file_path in glob.glob(os.path.join(directory, f"*.{extension}")):
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


def generate_pattern(level="number"):
    match level:
        case "number":
            base_dir = "number_patterns"
            os.makedirs(base_dir, exist_ok=True)

            for number in map(str, range(0, 10)):
                current_dir = os.path.join(base_dir, number)
                os.makedirs(current_dir, exist_ok=True)

                for code in range(1):
                    current_file = os.path.join(current_dir, str(code))

                    tex_file = f"{current_file}.tex"
                    log_file = f"{current_file}-dvi.log"
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

                    # dvitype my_symbols.dvi > symbols_positions.txt
                    log_file_path = os.path.join(current_dir, os.path.basename(log_file))
                    with open(log_file_path, "w") as log_f:
                        log_result = subprocess.run(
                            ["dvitype", os.path.basename(dvi_file)],
                            stdout=log_f,  # Redirect output to the log file
                            stderr=subprocess.PIPE,
                            cwd=current_dir,
                        )
                    with open(log_file_path, "r") as log_f:
                        dvi_log_content = log_f.read()

                    # Parse the extracted log content
                    # print(parse_dvi_log(dvi_log_content))
                    if tex_result.returncode != 0:
                        print(
                            f"Error creating log {log_file}: {log_result.stderr.decode()}"
                        )
                        continue

                    # Clean up
                    # for ext in ["aux", "log", "dvi", "tex"]:
                    #     delete_files_with_extension(current_dir, ext)

        case _:
            raise Exception("Unsupported pattern level")


def generate_dataset(level="number", count=42, seed=42):
    random.seed(seed)
    match level:
        case "number":
            base_dir = "dataset"
            os.makedirs(base_dir, exist_ok=True)

            json

            unique_numbers = random.sample(range(1, 10001), count)
            for code in range(count):
                current_file = os.path.join(base_dir, "%d_%s" % (code, level))

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
                    cwd=base_dir,
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
                    cwd=base_dir,
                )
                if dvi_result.returncode != 0:
                    print(
                        f"Error converting {dvi_file} to PNG: {dvi_result.stderr.decode()}"
                    )
                    continue

                # Clean up
                for ext in ["aux", "log", "dvi", "tex"]:
                    delete_files_with_extension(base_dir, ext)

        case _:
            raise Exception("Unsupported pattern level")


# Run the function
generate_pattern()
# generate_dataset()
