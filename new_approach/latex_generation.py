import subprocess
import os
import glob
import json
import random


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
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def generate_pattern(level="number", count=1000):
    match level:
        case "number":
            base_dir = "number_patterns"
            os.makedirs(base_dir, exist_ok=True)

            # for number in map(str, range(0, 10)):
                # current_dir = os.path.join(base_dir, number)
                # os.makedirs(base_dir, exist_ok=True)
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
                    cwd=base_dir,
                )
                if dvi_result.returncode != 0:
                    print(
                        f"Error converting {dvi_file} to PNG: {dvi_result.stderr.decode()}"
                    )
                    continue

                # Clean up
                for ext in ["aux", "log", "dvi", "tex"]:
                    delete_files_with_extension(base_dir,ext)

        case _:
            raise Exception("Unsupported pattern level")



def generate_dataset(level="number", seed = 42):
    match level:
            case "number":
                base_dir = "dataset"
                os.makedirs(base_dir, exist_ok=True)

                for number in map(str, range(0, 10)):
                    current_dir = os.path.join(base_dir, number)
                    os.makedirs(current_dir, exist_ok=True)

                    for code in range(1, 2):
                        current_file = os.path.join(current_dir, str(code))

                        tex_file = f"{current_file}.tex"
                        with open(tex_file, "w") as f:
                            f.write(latex % number)

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

                        # Clean up
                        for ext in ["aux", "log", "dvi", "tex"]:
                            delete_files_with_extension(current_dir,ext)

            case _:
                raise Exception("Unsupported pattern level")


# Run the function
generate_pattern()
