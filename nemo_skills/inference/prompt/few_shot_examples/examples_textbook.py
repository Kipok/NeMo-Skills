examples_detailed = [
    {"question": "$\\frac{\\left(152\\frac{3}{4}-148\\frac{3}{8}\\right)\\cdot 0.3}{0.2}$",
     "expected_answer": "6.5625",
     "generated_solution": """
To solve the expression $\\frac{\\left(152\\frac{3}{4}-148\\frac{3}{8}\\right)\\cdot 0.3}{0.2}$, we can follow these steps:

1. Convert the mixed numbers to improper fractions.
2. Subtract the fractions.
3. Multiply the result by $0.3$.
4. Divide by $0.2$.

### Step 1: Convert the mixed numbers to improper fractions

For $152\\frac{3}{4}$:
$ 152\\frac{3}{4} = 152 + \\frac{3}{4} = \\frac{152 \\cdot 4 + 3}{4} = \\frac{608 + 3}{4} = \\frac{611}{4} $

For $148\\frac{3}{8}$:
$ 148\\frac{3}{8} = 148 + \\frac{3}{8} = \\frac{148 \\cdot 8 + 3}{8} = \\frac{1184 + 3}{8} = \\frac{1187}{8} $

### Step 2: Subtract the fractions

To subtract $\\frac{611}{4}$ and $\\frac{1187}{8}$, we need a common denominator:
$ \\frac{611}{4} = \\frac{611 \\cdot 2}{4 \\cdot 2} = \\frac{1222}{8} $

Now, subtract the fractions:
$ \\frac{1222}{8} - \\frac{1187}{8} = \\frac{1222 - 1187}{8} = \\frac{35}{8} $

### Step 3: Multiply the result by $0.3$

$ \\frac{35}{8} \\cdot 0.3 = \\frac{35 \\cdot 0.3}{8} = \\frac{10.5}{8} $

### Step 4: Divide by $0.2$

$ \\frac{\\frac{10.5}{8}}{0.2} = \\frac{10.5}{8 \\cdot 0.2} = \\frac{10.5}{1.6} $

Simplify the division:
$ \\frac{10.5}{1.6} = \\frac{10.5 \\cdot 10}{1.6 \\cdot 10} = \\frac{105}{16} $

So, the final answer is:
$ \\boxed{\\frac{105}{16}} $
""".strip()
    },
    {"question": "$\\frac{172\\frac{5}{6}-170\\frac{1}{3}+3\\frac{5}{12}}{0.8\\cdot 0.25}$", "original_answer": "$29\\frac{7}{12}$", "expected_answer": "29\\frac{7}{12}",
     "generated_solution": """
$ \\frac{172\\frac{5}{6} - 170\\frac{1}{3} + 3\\frac{5}{12}}{0.8 \\cdot 0.25} $

### Step 1: Convert the mixed numbers to improper fractions

For $172\\frac{5}{6}$:
$ 172\\frac{5}{6} = 172 + \\frac{5}{6} = \\frac{172 \\cdot 6 + 5}{6} = \\frac{1032 + 5}{6} = \\frac{1037}{6} $

For $170\\frac{1}{3}$:
$ 170\\frac{1}{3} = 170 + \\frac{1}{3} = \\frac{170 \\cdot 3 + 1}{3} = \\frac{510 + 1}{3} = \\frac{511}{3} $

For $3\\frac{5}{12}$:
$ 3\\frac{5}{12} = 3 + \\frac{5}{12} = \\frac{3 \\cdot 12 + 5}{12} = \\frac{36 + 5}{12} = \\frac{41}{12} $

### Step 2: Perform the operations on the fractions

To perform the operations, we need a common denominator. The least common multiple (LCM) of 6, 3, and 12 is 12.

Convert each fraction to have a denominator of 12:
$ \\frac{1037}{6} = \\frac{1037 \\cdot 2}{6 \\cdot 2} = \\frac{2074}{12} $
$ \\frac{511}{3} = \\frac{511 \\cdot 4}{3 \\cdot 4} = \\frac{2044}{12} $
$ \\frac{41}{12} \\text{ (already has denominator 12)} $

Now, perform the operations:
$ \\frac{2074}{12} - \\frac{2044}{12} + \\frac{41}{12} = \\frac{2074 - 2044 + 41}{12} = \\frac{71}{12} $

### Step 3: Multiply $0.8$ and $0.25$ and divide the result from step 2 by this product

First, compute the product:
$ 0.8 \\cdot 0.25 = 0.2 $

Now, divide the fraction by $0.2$:
$ \\frac{\\frac{71}{12}}{0.2} = \\frac{71}{12 \\cdot 0.2} = \\frac{71}{2.4} $

Simplify the division:
$ \\frac{71}{2.4} = \\frac{71 \\cdot 10}{2.4 \\cdot 10} = \\frac{710}{24} $

Simplify the fraction:
$ \\frac{710}{24} = \\frac{355}{12} $

So, the final answer is:
$ \\boxed{\\frac{355}{12}} $
""".strip()
    },
    {"question": "$\\frac{215\\frac{9}{16}-208\\frac{3}{4}+\\frac{1}{2}}{0.0001:0.005}$", "original_answer": "$365\\frac{5}{8}$", "expected_answer": "365\\frac{5}{8}",
     "generated_solution": """
To solve the expression $\\frac{215\\frac{9}{16} - 208\\frac{3}{4} + \\frac{1}{2}}{0.0001 \\div 0.005}$, we will follow these steps:

1. Convert the mixed numbers to improper fractions.
2. Perform the operations (subtraction and addition) on the fractions.
3. Simplify the division in the denominator.
4. Divide the result from step 2 by the simplified denominator.

### Step 1: Convert the mixed numbers to improper fractions

For $215\\frac{9}{16}$:
$ 215\\frac{9}{16} = 215 + \\frac{9}{16} = \\frac{215 \\cdot 16 + 9}{16} = \\frac{3440 + 9}{16} = \\frac{3449}{16} $

For $208\\frac{3}{4}$:
$ 208\\frac{3}{4} = 208 + \\frac{3}{4} = \\frac{208 \\cdot 4 + 3}{4} = \\frac{832 + 3}{4} = \\frac{835}{4} $

For $\\frac{1}{2}$:
$ \\frac{1}{2} \\text{ (already in fractional form)} $

### Step 2: Perform the operations on the fractions

To perform the operations, we need a common denominator. The least common multiple (LCM) of 16, 4, and 2 is 16.

Convert each fraction to have a denominator of 16:
$ \\frac{3449}{16} \\text{ (already has denominator 16)} $
$ \\frac{835}{4} = \\frac{835 \\cdot 4}{4 \\cdot 4} = \\frac{3340}{16} $
$ \\frac{1}{2} = \\frac{1 \\cdot 8}{2 \\cdot 8} = \\frac{8}{16} $

Now, perform the operations:
$ \\frac{3449}{16} - \\frac{3340}{16} + \\frac{8}{16} = \\frac{3449 - 3340 + 8}{16} = \\frac{117}{16} $

### Step 3: Simplify the division in the denominator

Simplify $0.0001 \\div 0.005$:
$ 0.0001 \\div 0.005 = 0.0001 \\div 0.005 = 0.0001 \\times \\frac{1}{0.005} = 0.0001 \\times 200 = 0.02 $

### Step 4: Divide the result from step 2 by the simplified denominator

Now, divide the fraction by $0.02$:
$ \\frac{\\frac{117}{16}}{0.02} = \\frac{117}{16 \\cdot 0.02} = \\frac{117}{0.32} $

Simplify the division:
$ \\frac{117}{0.32} = \\frac{117 \\cdot 100}{0.32 \\cdot 100} = \\frac{11700}{32} $

Simplify the fraction:
$ \\frac{11700}{32} = \\frac{11700 \\div 4}{32 \\div 4} = \\frac{2925}{8} $

So, the final answer is:
$\\boxed{\\frac{2925}{8}}$
""".strip()
    },
]


examples_short = [
    {
        "question": "$\\frac{\\left(152\\frac{3}{4}-148\\frac{3}{8}\\right)\\cdot 0.3}{0.2}$",
        "expected_answer": "6.5625",
        "generated_solution": """
$
\\frac{\\left(152\\frac{3}{4}-148\\frac{3}{8}\\right)\\cdot 0.3}{0.2}
$

$
=\\frac{\\left(152.75 - 148.375\\right)\\cdot 0.3}{0.2}
$

$
=\\frac{4.375\\cdot 0.3}{0.2}
$

$
=\\frac{1.3125}{0.2}
$

$ = 6.5625 $

Thus, the final result is:

$\\boxed{6.5625}$
""".strip()
    },
    {
        "question": "$\\frac{172\\frac{5}{6}-170\\frac{1}{3}+3\\frac{5}{12}}{0.8\\cdot 0.25}$",
        "original_answer": "$29\\frac{7}{12}$",
        "expected_answer": "29\\frac{7}{12}",
        "generated_solution": """
$
\\frac{172\\frac{5}{6}-170\\frac{1}{3}+3\\frac{5}{12}}{0.8\\cdot 0.25}
$

$
=\\frac{172.8333 - 170.3333 + 3.4167}{0.8\\cdot 0.25}
$

$
=\\frac{172.8333 - 170.3333 + 3.4167}{0.2}
$

$
=\\frac{5.9167}{0.2}
$

$
= 29.5835
$

Thus, the final result is:

$\\boxed{29.5835}$
""".strip()
    },
    {"question": "$\\frac{215\\frac{9}{16}-208\\frac{3}{4}+\\frac{1}{2}}{0.0001:0.005}$", "original_answer": "$365\\frac{5}{8}$", "expected_answer": "365\\frac{5}{8}",
     "generated_solution": """
$
\\frac{215\\frac{9}{16}-208\\frac{3}{4}+\\frac{1}{2}}{0.0001\\div 0.005}
$

$
=\\frac{215.5625 - 208.75 + 0.5}{0.0001\\div 0.005}
$

$
=\\frac{215.5625 - 208.75 + 0.5}{0.02}
$

$
=\\frac{7.3125}{0.02}
$

$
= 365.625
$

Thus, the final result is:

$\\boxed{365.625}$   
""".strip()
    },
]

examples_short_precise = [
    {"question": "$\\frac{\\left(152\\frac{3}{4}-148\\frac{3}{8}\\right)\\cdot 0.3}{0.2}$",
     "expected_answer": "6.5625",
     "generated_solution": """
$
\\frac{\\left(152\\frac{3}{4}-148\\frac{3}{8}\\right)\\cdot 0.3}{0.2}
$

$
=\\frac{\\left(\\frac{611}{4} - \\frac{1187}{8}\\right)\\cdot 0.3}{0.2}
$

$
=\\frac{\\left(\\frac{1222}{8} - \\frac{1187}{8}\\right)\\cdot 0.3}{0.2}
$

$
=\\frac{\\frac{35}{8}\\cdot \\frac{3}{10}}{0.2}
$

$
=\\frac{\\frac{105}{80}}{0.2}
$

$
=\\frac{105}{80}} \\cdot 5
$

$
=\\frac{105}{16}
$

Thus, the final result is:

$\\boxed{\\frac{105}{16}}$
""".strip()
    },
    {"question": "$\\frac{172\\frac{5}{6}-170\\frac{1}{3}+3\\frac{5}{12}}{0.8\\cdot 0.25}$", "original_answer": "$29\\frac{7}{12}$", "expected_answer": "29\\frac{7}{12}",
     "generated_solution": """
$
\\frac{172\\frac{5}{6}-170\\frac{1}{3}+3\\frac{5}{12}}{0.8\\cdot 0.25}
$

$
=\\frac{\\frac{1037}{6}-\\frac{511}{3}+\\frac{41}{12}}{0.8\\cdot 0.25}
$

$
=\\frac{\\frac{2074}{12}-\\frac{2044}{12}+\\frac{41}{12}}{0.8\\cdot 0.25}
$

$
=\\frac{\\frac{71}{12}}{0.8\\cdot 0.25}
$

$
=\\frac{\\frac{71}{12}}{0.2}
$

$
=\\frac{71}{12} \\cdot 5
$

$
=\\frac{355}{12}
$

Thus, the final result is:

$\\boxed{\\frac{355}{12}}$
""".strip()
    },
    {"question": "$\\frac{215\\frac{9}{16}-208\\frac{3}{4}+\\frac{1}{2}}{0.0001:0.005}$", "original_answer": "$365\\frac{5}{8}$", "expected_answer": "365\\frac{5}{8}",
     "generated_solution": """
$
\\frac{215\\frac{9}{16}-208\\frac{3}{4}+\\frac{1}{2}}{0.0001\\div 0.005}
$

$
=\\frac{\\frac{3449}{16}-\\frac{835}{4}+\\frac{1}{2}}{0.0001\\div 0.005}
$

$
=\\frac{\\frac{3449}{16}-\\frac{3340}{16}+\\frac{8}{16}}{0.0001\\div 0.005}
$

$
=\\frac{\\frac{117}{16}}{0.0001\\div 0.005}
$

$
=\\frac{\\frac{117}{16}}{0.02}
$

$
=\\frac{117}{16} \\cdot 50
$

$
=\\frac{5850}{16}
$

$
=\\frac{2925}{8}
$

Thus, the final result is:

$\\boxed{\\frac{2925}{8}}$
""".strip()
    },
]


examples_shorter_precise = [
    {
        "question": "$\\frac{\\left(152\\frac{3}{4}-148\\frac{3}{8}\\right)\\cdot 0.3}{0.2}$",
        "expected_answer": "6.5625",
        "generated_solution": """
$
\\frac{\\left(152\\frac{3}{4}-148\\frac{3}{8}\\right)\\cdot 0.3}{0.2}
$

$
=\\frac{\\left(\\frac{611}{4} - \\frac{1187}{8}\\right)\\cdot 3}{2}
$

$
=\\left(\\frac{1222 - 1187}{8}\\right)\\cdot \\frac{3}{2}
$

$
=\\frac{105}{16}
$

Thus, the final result is:

$\\boxed{\\frac{105}{16}}$
""".strip()
    },
    {
        "question": "$\\frac{172\\frac{5}{6}-170\\frac{1}{3}+3\\frac{5}{12}}{0.8\\cdot 0.25}$", "original_answer": "$29\\frac{7}{12}$", "expected_answer": "29\\frac{7}{12}",
        "generated_solution": """
$
\\frac{172\\frac{5}{6}-170\\frac{1}{3}+3\\frac{5}{12}}{0.8\\cdot 0.25}
$

$
=\\frac{\\frac{1037}{6}-\\frac{511}{3}+\\frac{41}{12}}{0.2}
$

$
=\\frac{2074 - 2044 + 41}{12}\\cdot 5
$

$
=\\frac{71}{12}\\cdot 5
$

$
=\\frac{355}{12}
$

Thus, the final result is:

$\\boxed{\\frac{355}{12}}$
""".strip()
    },
    {
        "question": "$\\frac{215\\frac{9}{16}-208\\frac{3}{4}+\\frac{1}{2}}{0.0001:0.005}$",
        "original_answer": "$365\\frac{5}{8}$", "expected_answer": "365\\frac{5}{8}",
        "generated_solution": """
$
\\frac{215\\frac{9}{16}-208\\frac{3}{4}+\\frac{1}{2}}{0.0001\\div 0.005}
$

$
=\\frac{\\frac{3449}{16}-\\frac{835}{4}+\\frac{1}{2}}{0.02}
$

$
=\\frac{3449 - 3340 + 8}{16}\\cdot 50
$

$
=\\frac{117}{8}\\cdot 25
$

$
=\\frac{2925}{8}
$

Thus, the final result is:

$\\boxed{\\frac{2925}{8}}$
""".strip()
    },
]

examples_map = {
    "examples_textbook_detailed": examples_detailed,
    "examples_textbook_short": examples_short,
    "examples_textbook_short_precise": examples_short_precise,
    "examples_textbook_shorter_precise": examples_shorter_precise,
}