# %%
import random

# Словарь fallback-значений для нетерминалов, чтобы на максимальной глубине ветка завершалась корректно.

fallback_dict = {
    "expr": "1",
    "sum": "1",
    "product": "1",
    "power": "1",
    "postfix": "1",
    "primary": "1",
    "group": r"{1}",
    "frac_expr": r"{1}{1}",
    "integral_limits": "",
    "expr_opt": "",
    "limit_limits": "",
}

# %%
def generate_formula(
    grammar,
    weights,
    terminal_generators,
    symbol="start",
    depth=0,
    max_depth=15,
    recursion_bonus=0.5,
    default_weight=0.9,
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
        return ""
    
    # Если мы на максимальной глубине и symbol – нетерминал, возвращаем fallback.
    if depth == max_depth and symbol in grammar:
        return fallback_dict.get(symbol, "")
    
    # Если symbol не является ключом грамматики, значит это терминал.
    if symbol not in grammar:
        if symbol in terminal_generators:
            return terminal_generators[symbol]()
        return symbol

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
    
    result = ""
    for token in chosen_alt:
        # Если токен – нетерминал, то если следующая глубина равна max_depth, используем fallback,
        # иначе продолжаем рекурсию.
        if token in grammar:
            if depth + 1 == max_depth:
                result += fallback_dict.get(token, "")
            else:
                result += generate_formula(
                    grammar,
                    weights,
                    terminal_generators,
                    token,
                    depth + 1,
                    max_depth,
                    recursion_bonus,
                    default_weight,
                )
        else:
            if token in terminal_generators:
                result += terminal_generators[token]()
            else:
                result += token
            
    return result

# %%

grammar = {
    "start": [["expr"]],
    "expr": [["sum"]],
    "sum": [
        ["product"],
        ["(", "sum", ")", "product"],
        ["(", "sum", ")", "-", "product"],
    ],
    "product": [
        ["power"],
        ["product", "\\cdot", " ", "power"],
        ["product", "\\times", " ", "power"],
        ["product", "/", " ", "power"],
        ["product", "BINOP_FUNC", " ", "power"],
    ],
    "power": [
        ["postfix"],
        ["postfix", "^", "{", "power", "}"],
    ],
    "postfix": [
        ["primary"],
        ["postfix", "_", "{", "primary", "}"],
    ],
    "primary": [
        ["NUMBER"],
        ["GREEK"],
        ["LATIN"],
        ["FRAC", "frac_expr"],
        ["BINOP_FUNC", "group"],
        ["FUNCTION", "group"],
        ["(", "expr", ")"],
        ["[", "expr", "]"],
        ["{", "expr", "}"],
        ["INTEGRAL", "integral_limits", " ", "expr_opt"],
        ["LIMIT", "limit_limits", " ", "expr"],
    ],
    "group": [["{", "expr", "}"]],
    "frac_expr": [["{", "expr", "}", "{", "expr", "}"]],
    "integral_limits": [
        [],
        ["_", "group"],
        ["^", "group"],
        ["_", "group", "^", "group"],
    ],
    "expr_opt": [
        [],
        ["expr"],
    ],
    "limit_limits": [
        [],
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
}

terminal_generators = {
    "BINOP_FUNC": lambda: random.choice(["\\min", "\\max", "\\opt"]),
    "FUNCTION": lambda: random.choice(
        ["\\sin", "\\cos", "\\tan", "\\log", "\\ln", "\\exp"]
    ),
    "FRAC": lambda: "\\frac",
    "NUMBER": lambda: str(random.randint(1, 9)),
    "GREEK": lambda: random.choice(
        ["\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon"]
    ),
    "LATIN": lambda: "".join(
        random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(1, 4))
    ),
    "INTEGRAL": lambda: "\\int",
    "LIMIT": lambda: "\\lim",
}


# %%
random.seed(42)
# Генерация нескольких формул с фиксированной глубиной
n = 10000
formulas = [
    generate_formula(grammar, weights, terminal_generators)
    for _ in range(n)
]
for formula in formulas:
    print(formula)

# %%



