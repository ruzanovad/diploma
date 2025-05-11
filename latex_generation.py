import subprocess
import os
import glob
import random
from PIL import Image
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor  # for concurrency
from functools import partial  # for concurrency
from collections import Counter, defaultdict
import utils
import yaml
import numpy as np

random.seed(42)
load_dotenv()

latex = r"""\documentclass{standalone}[border={10pt 10pt 10pt 10pt}]
\usepackage{amsmath}
\begin{document}
%s
\end{document}
"""


terminal_generators = {
    "BINOP_FUNC": lambda: (x := random.choice(["\\min", "\\max"]), {x: 1}),
    "FUNCTION": lambda: (
        x := random.choice(
            [
                "\\sin",
                "\\cos",
                "\\tan",
                "\\log",
                "\\ln",
                "\\exp",
                "\\sqrt",
                "\\arcsin",
                "\\arccos",
                "\\arctan",
            ]
        ),
        {x: 1},
    ),
    "FRAC": lambda: ("\\frac", {"\\frac": 1}),
    "NUMBER": lambda: (
        x := (
            str(random.randint(1, 9))
            if random.random() < 0.3
            else f"{random.randint(0, 9999)}.{random.randint(0, 9999)}"
        ),
        dict(Counter(x)),
    ),
    "GREEK": lambda: (
        x := random.choice(
            [
                "\\alpha",
                "\\beta",
                "\\gamma",
                "\\delta",
                "\\epsilon",
                "\\zeta",
                "\\eta",
                "\\phi",
                "\\kappa",
                "\\lambda",
                "\\mu",
                "\\nu",
                "\\xi",
                "\\pi",
                "\\rho",
                "\\sigma",
                "\\tau",
                "\\varepsilon",
                "\\phi",
                "\\varphi",
                "\\chi",
                "\\psi",
                "\\omega",
            ]
        ),
        {x: 1},
    ),
    "LATIN": lambda: (
        x := "".join(
            random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(1, 4))
        ),
        dict(Counter(x)),
    ),
    "CAPS_LATIN": lambda: (
        x := "".join(
            random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=random.randint(1, 4))
        ),
        dict(Counter(x)),
    ),
    "INTEGRAL": lambda: ("\\int", {"\\int": 1}),
    "SUMMARY": lambda: ("\\sum", {"\\sum": 1}),
    "PROD": lambda: ("\\prod", {"\\prod": 1}),
    "LIMIT": lambda: ("\\lim", {"\\lim": 1}),
}  # Возвращает случайный терминал из заданного списка вместе со словарем частот.


def get_terminal():
    return terminal_generators[
        random.choice(
            [
                "BINOP_FUNC",
                "FUNCTION",
                "FRAC",
                "NUMBER",
                "GREEK",
                "LATIN",
                "CAPS_LATIN",
                "INTEGRAL",
                "LIMIT",
                "SUMMARY",
                "PROD",
            ]
        )
    ]()


def merge_freqs(items):
    """Объединяет словари частот из списка кортежей (значение, словарь)."""
    merged = Counter()
    values = []
    for val, freq in items:
        values.append(val)
        merged.update(freq)
    return values, dict(merged)


# Словарь fallback-значений для нетерминалов, чтобы на максимальной глубине ветка завершалась корректно.
fallback_dict = {
    "expr": (
        merge_freqs([("-", {"-": 1}), get_terminal()])
        if random.random() < 0.5
        else merge_freqs([get_terminal()])
    ),
    "sum": merge_freqs([get_terminal(), ("+", {"+": 1}), get_terminal()]),
    "product": merge_freqs([get_terminal()]),
    "power": merge_freqs([get_terminal()]),
    "postfix": merge_freqs([get_terminal()]),
    "primary": merge_freqs([get_terminal()]),
    "group": merge_freqs([("{", {"{": 1}), get_terminal(), ("}", {"}": 1})]),
    "frac_expr": merge_freqs(
        [
            ("{", {"{": 1}),
            get_terminal(),
            ("}", {"}": 1}),
            ("{", {"{": 1}),
            get_terminal(),
            ("}", {"}": 1}),
        ]
    ),
    "integral_limits": merge_freqs([get_terminal()]),
    "expr_opt": merge_freqs([get_terminal()]),
    "limit_limits": merge_freqs(
        [
            terminal_generators["LATIN"](),
            ("=", {"=": 1}),
            (" ", {" ": 1}),
            get_terminal(),
        ]
    ),
}

grammar = {
    "start": [["expr"]],
    "expr": [["sum"]],
    "sum": [
        ["product"],
        ["(", "sum", ")", "product"],
        ["(", "sum", ")", "-", "(", "product", ")"],
    ],
    "product": [
        ["power"],
        ["product", "\\cdot", "power"],
        ["product", "\\times", "power"],
        ["product", "/", "power"],
        ["product", "BINOP_FUNC", "power"],
    ],
    "power": [
        ["{", "postfix", "}"],
        ["{", "postfix", "}", "^", "{", "power", "}"],
    ],
    "postfix": [
        ["{", "primary", "}"],
        ["{", "postfix", "}", "_", "{", "primary", "}"],
    ],
    "primary": [
        ["NUMBER"],
        ["GREEK"],
        ["LATIN"],
        ["CAPS_LATIN"],
        ["FRAC", "frac_expr"],
        ["BINOP_FUNC", "group"],
        ["FUNCTION", "group"],
        ["(", "expr", ")"],
        ["[", "expr", "]"],
        ["{", "expr", "}"],
        ["INTEGRAL", "integral_limits", "expr_opt"],
        ["SUMMARY", "integral_limits", "expr_opt"],
        ["PROD", "integral_limits", "expr_opt"],
        ["LIMIT", "limit_limits", "expr"],
        ["|", "expr", "|"],
    ],
    "group": [["{", "expr", "}"]],
    "frac_expr": [["{", "expr", "}", "{", "expr", "}"]],
    "integral_limits": [
        ["_", "group"],
        ["^", "group"],
        ["_", "group", "^", "group"],
    ],
    "expr_opt": [
        ["expr"],
    ],
    "limit_limits": [
        ["_", "group"],
    ],
}

weights = {
    "+": 1.0,
    "-": 1.0,
    "\\cdot": 1.0,
    "/": 1.0,
    "BINOP_FUNC": 1.0,
    "^": 1.0,
    "_": 1.0,
    "LIMIT": 1.0,
    "INTEGRAL": 1.0,
    "FUNCTION": 1.0,
    "FRAC": 1.0,
    "SUMMARY": 1.0,
    "PROD": 1.0,
}


def generate_formula(
    grammar,
    weights,
    terminal_generators,
    symbol="start",
    depth=0,
    max_depth=10,
    recursion_bonus=0.5,
    default_weight=0.8,
):
    """
    Рекурсивная генерация формулы с фиксированной максимальной глубиной.
    Если достигнута максимальная глубина, для нетерминалов возвращаются
    заранее заданные fallback-значения, чтобы ветки завершались терминалами.

    :param grammar: словарь грамматики, где ключ – нетерминал, а значение – список альтернатив (каждая альтернатива – список токенов)
    :param weights: словарь весов для операторов и спец-токенов
    :param terminal_generators: словарь генераторов для терминальных символов
    :param symbol: текущий символ (нетерминал или терминал)
    :param depth: текущая глубина рекурсии
    :param max_depth: максимальная допустимая глубина рекурсии
    :param recursion_bonus: вес для рекурсивного вызова (если текущий символ встречается внутри своей же альтернативы)
    :param default_weight: вес по умолчанию для токенов, отсутствующих в weights
    :return: сгенерированная строка (формула)
    """
    # Если глубина превышает max_depth – возвращаем пустую строку.
    if depth > max_depth:
        return [], {}

    # Если мы на максимальной глубине и symbol – нетерминал, возвращаем fallback.
    if depth == max_depth and symbol in grammar:
        token, terminals = fallback_dict.get(symbol, "")
        return token, terminals

    # Если symbol не является ключом грамматики, значит это терминал.
    if symbol not in grammar:
        if symbol in terminal_generators:
            token, freqs = terminal_generators[symbol]()
            return [token], freqs
        else:
            token = symbol
            return [token], {token: 1}

    alternatives = grammar[symbol]
    alt_weights = []
    for alt in alternatives:
        total_weight = 0
        for token in alt:
            if token == symbol:
                effective = recursion_bonus if depth < max_depth else default_weight
            else:
                effective = weights.get(token, default_weight)
            total_weight += effective
        alt_weights.append(total_weight)

    # Инвертированные веса: чем меньше сумма весов, тем выше вероятность выбора
    epsilon = 1e-6
    inv_weights = [1 / (w + epsilon) for w in alt_weights]
    total_inv = sum(inv_weights)
    probabilities = [w / total_inv for w in inv_weights]

    chosen_alt = random.choices(alternatives, weights=probabilities, k=1)[0]

    total_counts = {}
    result = []
    for token in chosen_alt:
        # Если токен – нетерминал, то если следующая глубина равна max_depth, используем fallback,
        # иначе продолжаем рекурсию.
        if token in grammar:
            if depth + 1 == max_depth:
                fallback, terminals = fallback_dict.get(token, "")
                result += fallback
                for k, v in terminals.items():
                    total_counts[k] = total_counts.get(k, 0) + v
            else:
                sub_formula, sub_counts = generate_formula(
                    grammar,
                    weights,
                    terminal_generators,
                    token,
                    depth + 1,
                    max_depth,
                    recursion_bonus,
                    default_weight,
                )
                result += sub_formula
                for k, v in sub_counts.items():
                    total_counts[k] = total_counts.get(k, 0) + v
        else:
            if token in terminal_generators:
                value, dictionary = terminal_generators[token]()
                result += [value]
                for k, v in dictionary.items():
                    total_counts[k] = total_counts.get(k, 0) + v
            else:
                result += [token]
                total_counts[token] = total_counts.get(token, 0) + 1

    return result, total_counts


def prepare_grammar_dataset_with_patterns(
    n,
    grammar,
    weights,
    terminal_generators,
    label,
    val=10,
    max_workers=10,
    verbose=100,
    min_length=30,
    max_length=70,
    test=10,
):
    """
    Генерация n формул с фиксированной глубиной.
    :param n: количество формул
    :param grammar: грамматика
    :param weights: веса
    :param terminal_generators: генераторы терминалов
    :return: список сгенерированных формул
    """
    print("Generating patterns:")
    generate_pattern(label, level="all")
    # d = defaultdict(int)

    print("Generating formulas:")
    assert val > 0
    assert test > 0
    assert verbose > 1

    base_dir = os.path.join("datasets", label, "dataset")
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")

    images_train_dir = os.path.join(images_dir, "train")
    images_val_dir = os.path.join(images_dir, "val")
    images_test_dir = os.path.join(images_dir, "test")

    labels_train_dir = os.path.join(labels_dir, "train")
    labels_val_dir = os.path.join(labels_dir, "val")
    labels_test_dir = os.path.join(labels_dir, "test")

    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    if val > 0:
        os.makedirs(images_val_dir, exist_ok=True)
        os.makedirs(labels_val_dir, exist_ok=True)
    if test > 0:
        os.makedirs(images_test_dir, exist_ok=True)
        os.makedirs(labels_test_dir, exist_ok=True)

    train_number = (n * (100 - val - test)) // 100
    val_number = (n * (val)) // 100
    test_number = n - train_number - val_number
    assert train_number + val_number + test_number == n

    symbols_dict = utils.load_symbols_from_templates(os.getenv("templates"))
    formulas = []
    # code, content, class_dict, images_dir, labels_dir, verbose, label = args

    counter = 0
    while counter < n:
        formula, dictionary = generate_formula(
            grammar,
            weights,
            terminal_generators,
            symbol="start",
            depth=0,
            max_depth=10,
            recursion_bonus=0.5,
            default_weight=0.8,
        )
        if min_length <= len(formula) <= max_length:
            counter += 1
            selected_classes = [
                x for x in symbols_dict.keys() if x in dictionary.keys()
            ]

            formulas.append(
                [
                    " ".join(formula),
                    {
                        key.strip(): symbols_dict[key.strip()]
                        for key in selected_classes
                    },
                ]
            )

    print("Saving formulas")

    # --- Generate the TRAIN set in parallel ---
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        train_args = [
            (
                i,
                *formulas[i],
                images_train_dir,
                labels_train_dir,
                verbose,
                label,
            )
            for i in range(train_number)
        ]
        executor.map(fill_file_job, train_args)

    # --- Generate the VAL set in parallel ---
    if val > 0:
        val_args = [
            (
                i,
                *formulas[i],
                images_val_dir,
                labels_val_dir,
                verbose,
                label,
            )
            for i in range(train_number, train_number + val_number)
        ]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            executor.map(fill_file_job, val_args)

    # --- Generate the TEST set in parallel ---
    if test > 0:

        test_args = [
            (
                i,
                *formulas[i],
                images_val_dir,
                labels_val_dir,
                verbose,
                label,
            )
            for i in range(train_number + val_number, n)
        ]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            executor.map(fill_file_job, test_args)

    # Finally clean up aux files
    for ext in ["aux", "log", "dvi"]:
        delete_files_with_extension(images_train_dir, ext)
        delete_files_with_extension(images_val_dir, ext)

    # Generate dataset.yaml
    utils.generate_yolo_yaml(
        os.getenv("templates"),
        os.path.join("datasets", label, "dataset.yaml"),
        label,
    )


def load_greek_letters(file_path):
    try:
        with open(file_path, "r") as file:
            list_of_letters = [line.strip() for line in file if line.strip()]
        # print(list_of_letters)
        return list_of_letters
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []


def delete_files_with_extension(directory, extension):
    for file_path in glob.glob(os.path.join(directory, f"*.{extension}")):
        try:
            os.remove(file_path)
            # print(f"Deleted: {file_path}")
        except Exception as e:
            pass
            # print(f"Error deleting {file_path}: {e}")


def generate_pattern(label, level="number"):
    base_dir = os.path.join("datasets", label, "patterns")

    os.makedirs(base_dir, exist_ok=True)
    match level:
        case "number":
            templates = utils.load_symbols_from_templates(
                os.getenv("templates"), all=False, files=["number.txt", "delimiter.txt"]
            )
        case "variable":
            templates = utils.load_symbols_from_templates(
                os.getenv("templates"),
                all=False,
                files=["number.txt", "delimiter.txt", "letter.txt", "greek-letter.txt"],
            )
        case "all":
            templates = utils.load_symbols_from_templates(
                os.getenv("templates"), all=True
            )
        case _:
            raise ValueError("Unknown level")

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
            for ext in ["aux", "log", "dvi"]:
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

    for ext in ["aux", "log", "dvi"]:
        delete_files_with_extension("", ext)


def fill_file_job(args):
    """
    Multiprocessing-compatible function.
    Receives a tuple (code, content, images_dir, labels_dir, verbose, print_every)
    """
    code, content, class_dict, images_dir, labels_dir, verbose, label = args
    try:

        fill_file(
            images_dir, labels_dir, code, content, class_dict, suffix="", label=label
        )

        if verbose > 1 and code % verbose == 0:
            print(f"[INFO] Processed {code} samples...")
    except Exception as e:
        print(f"Error processing {code}: {e}")


def fill_file(images_dir, labels_dir, code, content, class_dict: dict, suffix, label):

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

    bounding_boxes = utils.get_bounding_boxes(png_file, class_dict, label)

    with open(txt_file, "w") as file:
        for box in bounding_boxes:
            file.write(" ".join(box) + "\n")


def generate_one_with_label(images_dir, labels_dir, code, label, content: str):
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

    bounding_boxes = utils.get_bounding_boxes(png_file, label=label)

    with open(txt_file, "w") as file:
        for box in bounding_boxes:
            file.write(" ".join(box) + "\n")


def generate_number() -> str:
    return str(random.randint(0, 9999))


def generate_decimal(symbols_dict) -> str:
    integer_part = str(random.randint(0, 9999))
    decimal_part = str(random.randint(0, 9999))
    delim = random.choice([".", ","])
    classes = set(integer_part) | set(decimal_part) | set(delim)

    return f"{integer_part}{delim}{decimal_part}", {
        key.strip(): symbols_dict[key.strip()] for key in classes
    }


def generate_word(symbols_dict) -> str:
    length = random.randint(12, 17)
    text = "".join(
        random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", k=length)
    )
    return text, {key.strip(): symbols_dict[key.strip()] for key in text}


def generate_variable(symbols_dict) -> str:
    # 10 symbols, 1% of each class in average
    # for dataset of length equal to 3000 we have 300 symbols of each class
    # print(symbols_dict)
    l = random.choices([key for key in symbols_dict.keys()], k=15)
    classes = set(l)
    # choice = random.randint(0, 1)

    return " ".join(l), {key.strip(): symbols_dict[key.strip()] for key in classes}


def generate_greek(symbols_dict) -> str:
    global list_of_letters
    length = random.randint(9, 12)
    ch = random.choices(list_of_letters, k=length)
    greek = "".join(ch)
    classes = set(ch)
    return greek, {key.strip(): symbols_dict[key.strip()] for key in classes}


# def generate_dataset_only_templates():
#     """
#     for each class only one picture
#     """

#     base_dir = os.getenv("yolo_dataset_folder")
#     images_dir = os.path.join(base_dir, "images")
#     labels_dir = os.path.join(base_dir, "labels")

#     images_train_dir = os.path.join(images_dir, "train")
#     images_val_dir = os.path.join(images_dir, "val")
#     labels_train_dir = os.path.join(labels_dir, "train")
#     labels_val_dir = os.path.join(labels_dir, "val")

#     os.makedirs(images_train_dir, exist_ok=True)
#     os.makedirs(images_val_dir, exist_ok=True)
#     os.makedirs(labels_train_dir, exist_ok=True)
#     os.makedirs(labels_val_dir, exist_ok=True)

#     template_dir = os.getenv("templates")
#     symbols_dict = utils.load_symbols_from_templates(template_dir)
#     classes = {symbols_dict[key]: key for key in symbols_dict.keys()}

#     for code in range(len(classes)):
#         generate_one_with_label(images_train_dir, labels_train_dir, code, classes[code])

#     for code in range(len(classes), 2 * len(classes)):
#         fill_file(images_val_dir, labels_val_dir, code)

#     # Clean up
#     for ext in ["aux", "log", "dvi", "tex"]:
#         delete_files_with_extension(images_train_dir, ext)
#         delete_files_with_extension(images_val_dir, ext)

#     dataset_dir = os.path.basename(os.path.normpath(os.getenv("yolo_dataset_folder")))

#     yolo_config = {
#         "path": dataset_dir,
#         "train": "images/train",
#         "val": "images/val",
#         "nc": len(classes),
#         "names": classes,
#     }

#     with open("dataset.yaml", "w") as file:
#         yaml.dump(yolo_config, file, default_flow_style=False)


def generate_dataset(
    label,
    level="number",
    count=1000,
    seed=42,
    train=90,
    val=10,
    verbose=100,
):
    """
    Generates dataset in parallel using multiprocessing with optional verbose logging.
    """
    assert train + val == 100
    random.seed(seed)

    base_dir = os.path.join("datasets", label, "dataset")
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

    match level:
        case "number":
            symbols_dict = utils.load_symbols_from_templates(
                os.getenv("templates"), all=False, files=["number.txt", "delimiter.txt"]
            )
            train_args = [
                (
                    i,
                    *generate_decimal(symbols_dict),
                    images_train_dir,
                    labels_train_dir,
                    verbose,
                    label,
                )
                for i in range(train_number)
            ]

            val_args = [
                (
                    i,
                    *generate_decimal(symbols_dict),
                    images_val_dir,
                    labels_val_dir,
                    verbose,
                    label,
                )
                for i in range(train_number, train_number + val_number)
            ]
        case "variable":
            symbols_dict = utils.load_symbols_from_templates(
                template_dir=os.getenv("templates"),
                all=False,
                files=["number.txt", "delimiter.txt", "letter.txt", "greek-letter.txt"],
            )
            train_args = [
                (
                    i,
                    *generate_variable(symbols_dict),
                    images_train_dir,
                    labels_train_dir,
                    verbose,
                    label,
                )
                for i in range(train_number)
            ]

            val_args = [
                (
                    i,
                    *generate_variable(symbols_dict),
                    images_val_dir,
                    labels_val_dir,
                    verbose,
                    label,
                )
                for i in range(train_number, train_number + val_number)
            ]
        case "all":
            # Load classes from templates
            symbols_dict = utils.load_symbols_from_templates(os.getenv("templates"))
            classes = [
                key if len(key) == 1 else key + " " for key in symbols_dict.keys()
            ]
            lengths = np.random.randint(2, 20, count).tolist()

            def get_random_content(idx):
                """Генерирует случайное выражение и список использованных классов"""
                selected_classes = random.choices(classes, k=lengths[idx])
                expression = "".join(selected_classes)
                return expression, {
                    key.strip(): symbols_dict[key.strip()] for key in selected_classes
                }

            # Prepare argument lists for parallel execution
            train_args = [
                (
                    i,
                    *get_random_content(i),
                    images_train_dir,
                    labels_train_dir,
                    verbose,
                    label,
                )
                for i in range(train_number)
            ]

            val_args = [
                (
                    i,
                    *get_random_content(i),
                    images_val_dir,
                    labels_val_dir,
                    verbose,
                    label,
                )
                for i in range(train_number, train_number + val_number)
            ]
        case _:
            raise ValueError("Unknown level")

    # --- Generate the TRAIN set in parallel ---
    with ProcessPoolExecutor(max_workers=6) as executor:
        executor.map(fill_file_job, train_args)

    # --- Generate the VAL set in parallel ---
    with ProcessPoolExecutor(max_workers=6) as executor:
        executor.map(fill_file_job, val_args)

    # Finally clean up aux files
    for ext in ["aux", "log", "dvi"]:
        delete_files_with_extension(images_train_dir, ext)
        delete_files_with_extension(images_val_dir, ext)
    # Generate dataset.yaml
    match level:
        case "number":
            utils.generate_yolo_yaml(
                os.getenv("templates"),
                os.path.join("datasets", label, "dataset.yaml"),
                label,
                all=False,
                files=["number.txt", "delimiter.txt"],
            )
        case "variable":
            utils.generate_yolo_yaml(
                os.getenv("templates"),
                os.path.join("datasets", label, "dataset.yaml"),
                label,
                all=False,
                files=[
                    "number.txt",
                    "delimiter.txt",
                    "letter.txt",
                    "greek-letter.txt",
                ],
            )
        case "all":
            utils.generate_yolo_yaml(
                os.getenv("templates"),
                os.path.join("datasets", label, "dataset.yaml"),
                label,
            )
        case _:
            raise ValueError("Unknown level")


if __name__ == "__main__":
    # label = "number_margin_2000"
    # generate_pattern(level="number", label=label)
    # print("patterns done")
    # generate_dataset(count=2000, level="number", label=label)

    greek_letters_file = os.path.join("templates", "greek-letter.txt")
    list_of_letters = load_greek_letters(greek_letters_file)
    # label = "variable_margin_3000_uniform"
    # generate_pattern(level="variable", label=label)
    # print("patterns done")
    # generate_dataset(count=6000, level="variable", label=label)
    # formulas = []

    prepare_grammar_dataset_with_patterns(
        100000, grammar, weights, terminal_generators, "experiment"
    )
    # d = defaultdict(int)
    # for _ in range(10):
    #     x = generate_formula(grammar, weights, terminal_generators, max_depth=10)
    #     print(" ".join(x[0]))
    #     for k in x[1]:
    #         d[k] += 1
    # print(d)
