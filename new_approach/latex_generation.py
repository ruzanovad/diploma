import subprocess
import os
# import random
latex = r"""
\documentclass{standalone}
\usepackage{amsmath}
\begin{document}
%s
\end{document}
""" 


def generate_pattern(level="number", seed=42):
    # random.seed(seed)
    match level:
        case "number":
            if not os.path.exists("number_patterns"):
                os.mkdir("number_patterns")
            for number in map(str, range(0, 10)):
                current_dir = "number_patterns/%s" % number
                if not os.path.exists(current_dir):
                    os.mkdir(current_dir)
                for code in range(1, 2):
                    current_file = "%s/%d" % (current_dir, code)
                    with open("%s.tex" % current_file, "w") as f:
                        f.write(latex % number)
                    # tex -> dvi
                    subprocess.run(["latex", "-interaction=nonstopmode", "%s.tex"%current_file])
                    # dvi -> png
                    subprocess.run(["dvipng", "-T", "tight", "-D", "300", "-o", "%s.png" % current_file, "%s.dvi" % current_file])
                
        case _:
            raise Exception()

generate_pattern()