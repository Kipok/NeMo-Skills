tuple_extraction = [
        {
        "original_answer": "599.3",
        "generation": """
        599.3
""".strip(),
    },
    {
        "original_answer": "x=17, y=2\\frac{1}{4}$ of the weight of the round.",
        "generation": """
        (17, 2)

""".strip(),
    },
    {
        "original_answer": "(1) $x\\approx 1.86,\\ y\\approx -4.84$; (2) $x\\approx -1.86\\ y\\approx 4.84$;  (3) $x\\approx 4.84i,\\ y\\approx 1.86i$; (4) $x\\approx -4.84i,\\ y\\approx -1.86i$;  (5) $x=5,\\ y=4$; (6) $x=-5,\\ y=-4$;  (7) $x=4i,\\ y=-5i$; (8) $x=-4i,\\  y=5i$.",
        "generation": """
        ((1.86, -4.84), (-1.86, 4.84), (4.84i, 1.86i), (-4.84i, -1.86i), (5, 4), (-5, -4), (4i, -5i), (-4i, 5i))

""".strip(),
    },
    {
        "original_answer": "if $a\\ge 0$, then $x_1=0$, $x_2=\\frac{3}{4}a$; if $a<0$, the equation has no solution.",
        "generation": r"""
            \[
                x =
                \begin{cases}
                0 \text{ or } \frac{3}{4}a & \text{if } a \ge 0 \\
                \text{no solution} & \text{if } a < 0
                \end{cases}
            \]

""".strip(),
    },
    {
        "original_answer": "$5 \\le x \\le 10$",
        "generation": """
        5 \\le x \\le 10
""".strip(),
    },
]

examples_map = {
    "antonov_tuple_extraction": tuple_extraction,
}
