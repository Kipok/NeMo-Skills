# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from nemo_skills.prompt.utils import get_prompt


def test_generic_math_problem_augmentation_prompt():
    prompt = get_prompt('generic/problem-augmentation', 'llama3-instruct', 'math_problem_augmentation')

    expected_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

Write a new math problem inspired by a given one. Make the new problem reasonable and solvable.

Here are some examples of how to complete this task.

Original problem:
In the equation
$$5x^2-kx+1=0$$
determine $k$ such that the difference of the roots be equal to unity.

New problem:
The roots $x_1$ and $x_2$ of the equation
$$x^2-3ax+a^2=0$$
are such that
$x_1^2+x_2^2=1.75$.
Determine $a$.





Original problem:
Solve the following equation
$\\ds\\f{3+x}{3x}=\\sqrt {\\ds\\f{1}{9}+\\ds\\f{1}{x}\\sqrt {\\ds\\f{4}{9}+\\ds\\f{2}{x^2}}}$

New problem:
Solve the following equation
$\\sqrt {1+x\\sqrt {x^2+24}}=x+1$





Original problem:
In an infinitely decreasing geometric progression the sum
of all the terms occupying odd places is equal to 36, and that of all
the terms at even places equals 12.
Find the progression.

New problem:
The sum of the terms of an infinitely decreasing geometric
progression is equal to 56, and the sum of the squared terms of the
same progression is 448.
Find the first term and the common ratio.





Original problem:
Two railway stations are at a distance of 96 km from each other.
One train covers this distance 40 minutes faster than does the other.
The speed of the first train is 12 km/h higher than that of the second.
Determine the speed of both trains.

New problem:
A student was asked to multiply 78 by a two-digit number
in which the tens digit was three times as large as the units digit;
by mistake, he interchanged the digits in the second factor and
thus obtained a product smaller than the true product by 2808.
What was the true product?





Original problem:
What's the meaning of life?

Write another problem inspired by this one.
Don't just change the numbers and context, but try to create a problem that requires another approach to solve.
Start directly with the problem statement and DO NOT include any phrases such as "Here is a new math problem inspired by a given one".
After the problem is completed finish your response right away.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    assert prompt.fill({'problem': "What's the meaning of life?"}) == expected_prompt


def test_generic_gsm8k_problem_augmentation_prompt():
    prompt = get_prompt('generic/problem-augmentation-similar', 'nemotron-instruct', 'gsm8k_problem_augmentation')

    expected_prompt = """<extra_id_0>System

<extra_id_1>User
Write a new math problem similar to a given one. Make the new problem reasonable and solvable.

Here are some examples of how to complete this task.

Original problem:
Olivia has $23. She bought five bagels for $3 each. How much money does she have left?

New problem:
Aiden has $35. He purchased eight pencils for $2 each and a notebook for $5. How much money does he have remaining?





Original problem:
Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?

New problem:
Sarah collected 72 seashells during her beach vacation. On Thursday, she gave 15 seashells to her friend as a souvenir. On Friday, she found 8 more seashells while exploring the shore. How many seashells did Sarah have at the end of Friday?





Original problem:
Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?

New problem:
Samantha and David are preparing for their upcoming science fair project. They have four different experiments to conduct and a research paper to write. Each experiment is estimated to take 2 hours, and the research paper will require 8 hours to complete. To stay focused and productive, they plan to take a 15-minute break for every 1.5 hours of work and have three 20-minute snack breaks each day. Additionally, they allocate 45 minutes for lunch each day. If they want to limit their daily study time to 5 hours, how many days should they plan to work on their project over the next two weeks?





Original problem:
Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?

New problem:
Tom has 50 marbles, and his friend Jerry has 65 marbles. If they decide to play a game and bet 20 marbles each, how many marbles will they have left in total after the game?





Original problem:
There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?

New problem:
In a garden, there were 12 flowers. Every morning for a week (from Monday to Sunday), 3 more flowers were planted. How many flowers are there in the garden now?





Original problem:
Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?

New problem:
Sarah had 35 marbles. She gave some marbles to her friend Emma. Now Sarah has 18 marbles left. How many marbles did Sarah give to Emma?





Original problem:
Sam bought a dozen boxes, each with 30 highlighter pens inside, for $10 each box. He rearranged five of these boxes into packages of six highlighters each and sold them for $3 per package. He sold the rest of the highlighters separately at the rate of three pens for $2. How much profit did he make in total, in dollars?

New problem:
Amy purchased 8 crates, each containing 24 colorful markers, for $12 per crate. She decided to create sets of 4 markers each and sell them for $2 per set. The remaining markers she sold individually at a rate of 5 markers for $3. Calculate the total profit Amy made, in dollars.





Original problem:
There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?

New problem:
In a garden, there are 25 rose bushes. The gardener plans to plant some more rose bushes today. After planting, there will be a total of 40 rose bushes in the garden. How many rose bushes will the gardener plant today?





Original problem:
What's the meaning of life?

Write another problem similar to this one.
Start directly with the problem statement and DO NOT include any phrases such as "Here is a new math problem similar to a given one".
After the problem is completed finish your response right away.
<extra_id_1>Assistant
"""
    assert prompt.fill({'problem': "What's the meaning of life?"}) == expected_prompt


def test_generic_codegen_prompt():
    prompt = get_prompt('generic/codegen')

    expected_prompt = [
        {'role': 'system', 'content': ''},
        {
            'role': 'user',
            'content': '''
Here is a problem for which you need to generate/complete code:
def 2_plus_2():
    """Write code to solve 2 + 2"""

Please continue to complete the function with python programming language. You are not allowed to modify the given code and do the completion only.

The solution should be in the following format:
```python
# Your code here
```'''.strip(),
        },
    ]
    assert prompt.fill({'question': 'def 2_plus_2():\n    """Write code to solve 2 + 2"""'}) == expected_prompt


def test_generic_default_prompt():
    prompt = get_prompt('generic/default')

    expected_prompt = [
        {'role': 'system', 'content': ''},
        {
            'role': 'user',
            'content': 'How are you?',
        },
    ]
    assert prompt.fill({'question': 'How are you?'}) == expected_prompt


def test_generic_math_prompt():
    prompt = get_prompt('generic/math', 'llama3-instruct')

    expected_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.

2 + 2 = ?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    assert prompt.fill({'problem': '2 + 2 = ?'}) == expected_prompt


def test_generic_math_prompt_code_examples():
    prompt = get_prompt('generic/math', 'llama3-base', 'math_text_with_code')

    expected_prompt = """<|begin_of_text|>Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.

Here are some examples of problems and solutions you can refer to.

Problem:
A parabola with equation $y=x^2+bx+c$ passes through the points $(-1,-11)$ and $(3,17)$. What is $c$?

Solution:
Let's write down an equation for the parabola and solve for $c$ using sympy.
<llm-code>
import sympy as sp

# define the symbols
x, y, b, c = sp.symbols('x y b c')

# define the parabola equation
parabola_eq = sp.Eq(y, x**2 + b*x + c)

# the parabola passes through the points (-1,-11) and (3,17)
# so we substitute these points into the parabola equation
point_1 = parabola_eq.subs({x: -1, y: -11})
point_2 = parabola_eq.subs({x: 3, y: 17})

# we now have two equations and two unknowns (b and c)
# we will solve for b and c
solutions = sp.solve((point_1,point_2), (b, c))
solutions[c]
</llm-code><llm-code-output>

completed
[stdout]
-7
[/stdout]</llm-code-output>

So c is \\boxed{-7}





Problem:
Let $f(x)$ be an odd function.  Is $f(f(x))$ even, odd, or neither?

Enter "odd", "even", or "neither".

Solution:
The definition of an odd function is that $f(-x) = -f(x)$.
Applying this to $f(f(-x))$ we get $f(f(-x)) = f(-f(x)) = -f(f(x))$.
Thus, $f(f(x))$ is an \\boxed{odd} function.





Problem:
At the 2007 Math Olympics, Team Canada won $17$ out of a possible $100$ medals. Which one of the following is closest to the fraction of medals that they won? $$
\\frac{1}{4} \\qquad \\frac{1}{5} \\qquad \\frac{1}{6} \\qquad \\frac{1}{7} \\qquad \\frac{1}{8}
$$

Solution:
Let's use sympy to print out the differences between the fraction of medals that Canada won and each of the options.
<llm-code>
from sympy import Rational, Abs

# team Canada won 17 out of 100 medals
medal_frac = Rational(17, 100)

# list of options
options = [Rational(1, 4), Rational(1, 5), Rational(1, 6), Rational(1, 7), Rational(1, 8)]

# let's print out the differences
[Abs(medal_frac - frac_option) for frac_option in options]
</llm-code><llm-code-output>

completed
[stdout]
[2/25, 3/100, 1/300, 19/700, 9/200]
[/stdout]</llm-code-output>

Let's now check which difference is the smallest.
<llm-code>
import numpy as np

# Calculate the idx of the closest option
min_idx = np.argmin([2/25, 3/100, 1/300, 19/700, 9/200])

# Print the closest option
print(options[min_idx])
</llm-code><llm-code-output>

completed
[stdout]
1/6
[/stdout]</llm-code-output>

So the answer is \\boxed{1/6}.





Problem:
A rectangular box $P$ is inscribed in a sphere of radius $r$. The surface area of $P$ is 384, and the sum of the lengths of its 12 edges is 112. What is $r$?

Solution:
Let the dimensions of $P$ be $x$, $y$, and $z$.
The sum of the box's edges is $4(x + y + z)$ and the surface area is $2xy + 2yz + 2xz$.
The diameter of the sphere is $2r$ and it's equal to the diagonal of the box.
Let's now write down the equations based on the above information and solve them using sympy.
<llm-code>
from sympy import symbols, Eq, solve

# define the variables
x, y, z, r = symbols('x y z r')

# equations based on the given information
eq1 = Eq(2 * (x*y + x*z + y*z), 384)
eq2 = Eq(4 * (x + y + z), 112)

# the diagonal of the box is the diameter of the sphere
diagonal_eq = Eq(x**2 + y**2 + z**2, (2*r)**2)

# solve the equations
solutions = solve((eq1, eq2, diagonal_eq), (x, y, z, r))

# let's see which values of r we get
[solution[-1] for solution in solutions]
</llm-code><llm-code-output>

completed
[stdout]
[-10, -10, 10, 10]
[/stdout]</llm-code-output>

Since the radius of the sphere is positive, we get $r = \\boxed{10}$.





Problem:
A bee starts flying from point $P_0$. She flies $1$ inch due east to point $P_1$. For $j \\ge 1$, once the bee reaches point $P_j$, she turns $30^{\\circ}$ counterclockwise and then flies $j+1$ inches straight to point $P_{j+1}$. When the bee reaches $P_{2015},$ how far from $P_0$ is she, in inches?

Solution:
We can represent the rotation action via $\\omega = e^{\\pi i/6}$.
Let's assume the bee starts at the origin, so $P_{2015}$ is at the point \\[z = 1 + 2 \\omega + 3 \\omega^2 + 4 \\omega^3 + \\dots + 2015 \\omega^{2014}.\\]
This is an arithmetic-geometric series which we can solve by simplifying the expression.
Alternatively, we can solve for |z| using sympy using the following code.
<llm-code>
from sympy import I, pi, exp, sqrt

# rotation action of 30 degrees
omega = exp(I * pi/6)

position = 0

for i in range(2015):
    delta = (i + 1) * omega**(i)
    position += delta

real, img = (position.as_real_imag())
# Distance from origin i.e. starting point
dist = sqrt(real**2 + img**2)
print(dist)
</llm-code><llm-code-output>

completed
[stdout]
sqrt(2)*(1008 + 1008*sqrt(3))
[/stdout]</llm-code-output>

So the bee is $\\boxed{1008\\sqrt{2} + 1008\\sqrt{6}}$ far from the starting point.





Here is the problem you need to solve:
2 + 2 = ?"""
    assert prompt.fill({'problem': '2 + 2 = ?'}) == expected_prompt


def test_generic_multichoice_prompt():
    prompt = get_prompt('generic/multichoice', 'default-base')

    expected_prompt = """Answer the following multiple choice question.
The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of A, B, C or D..

Question: How are you?

A. Good
B. Bad
C. And you?
D. 42

Think step by step before answering."""
    assert (
        prompt.fill({'question': 'How are you?', 'A': 'Good', 'B': 'Bad', 'C': 'And you?', 'D': '42'})
        == expected_prompt
    )


def test_llama3_instruct_math_prompt():
    prompt = get_prompt('llama3-instruct/math', 'llama3-instruct-nosys')

    expected_prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem.

Problem: 3 + 5?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    assert prompt.fill({'problem': '3 + 5?'}) == expected_prompt


def test_judge_arena():
    prompt = get_prompt('judge/arena')

    expected_prompt = [
        {
            'role': 'system',
            'content': '''
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]".
'''.strip(),
        },
        {
            'role': 'user',
            'content': '''
<|User Prompt|>
What's better for a cold: tea or coffee?

<|The Start of Assistant A's Answer|>
Tea for sure
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
I mean, coffee, why do you even ask?
<|The End of Assistant B's Answer|>
'''.strip(),
        },
    ]
    assert (
        prompt.fill(
            {
                'question': "What's better for a cold: tea or coffee?",
                "answer_1": "Tea for sure",
                "answer_2": "I mean, coffee, why do you even ask?",
            }
        )
        == expected_prompt
    )


def test_judge_math():
    prompt = get_prompt('judge/math')

    expected_prompt = [
        {
            'role': 'system',
            'content': '',
        },
        {
            'role': 'user',
            'content': '''
You will be asked to look at the two answers (predicted and expected) to a math problem and to judge whether they are equivalent within the context of the problem.

Please first explain your reasoning in a couple of sentences. Then respond with only Yes or No as your judgement on whether the two answers are the same.
When comparing answers only perform trivial simplifications.

Here are a few examples.


Example 1:
Problem: Factor $7x^3 - 21x^2 + 14x$.
Predicted answer: $7x(x - 2)(x - 1)$
Expected answer: $7x(x-1)(x-2)$

Reasoning: The order of the factors does not matter, so the answers are the same.
Judgement: Yes


Example 2:
Problem: A rectangle has a length of 6 meters and a width of 2 meters. If the length is reduced by 3 meters and the width is halved, what is the new area of the rectangle in square meters?
Predicted answer: 3/2
Expected answer: 1.5

Reasoning: 3/2 is the same as 1.5
Judgement: Yes


Example 3:
Problem: Simplify the expression $\\sqrt{7!}$, where $n!$ stands for $n\\cdot(n-1)\\cdot(n-2)\\cdots \\cdot 2\\cdot 1$.
Predicted answer: 71
Expected answer: 12\\sqrt{35}.

Reasoning: This is non-trivial to simplify, so the answers are different.
Judgement: No


Example 4:
Problem: What is the simplified form of the expression $\\sqrt{98 x^{3} y^{5} z} ?
\\begin{align*}
\\text{A)} & 2 x y z \\sqrt{7 x y z} &
\\text{B)} &  7 x^{2} y^{2} \\sqrt{2 y z}
\\\\
\\text{C)} & 7 x y^{2} \\sqrt{2 x y z}  &
\\text{D)} &49 x y^{2} \\sqrt{2 x y z}
\\\\
\\end{align*}
Predicted answer: 7 x y^{2} \\\\sqrt{2 x y z}
Expected answer: C

Reasoning: Predicted answer is the same as the expected answer choice C.
Judgement: Yes


Example 5:
Problem: A line segment of length $5$ has one endpoint at $(1, 2)$ and the other endpoint at $(4, b)$. Find all possible values of $b$, separated by commas.
Predicted answer:  -2, 6
Expected answer: 6, -2

Reasoning: The order doesn't matter in the context of the problem.
Judgement: Yes


Example 6:
Problem: Solve $\\tan x = \\sin x$ for $0 \\le x \\le 2 \\pi.$  Enter all the solutions, separated by commas.
Predicted answer: 0, \\pi
Expected answer: 0,\\pi,2\\pi.

Reasoning: Number of solutions is different.
Judgement: No


YOUR TASK

Problem: 1 + 1
Predicted answer: eh, 15?
Expected answer: 2
'''.strip(),
        },
    ]
    assert (
        prompt.fill(
            {
                'problem': "1 + 1",
                "predicted_answer": "eh, 15?",
                "expected_answer": "2",
            }
        )
        == expected_prompt
    )


def test_judge_check_contamination():
    prompt = get_prompt('judge/check-contamination')

    expected_prompt = [
        {
            'role': 'system',
            'content': '',
        },
        {
            'role': 'user',
            'content': '''
Help me determine if the following two math problems are the same.

First problem: 1 + 3
Second problem: what's 3 plus 1?

Disregard the names and minor changes in word order that appear within.
If the two problems are very similar and if they produce the same answer, we consider them to be the same problem.
Respond with only "True" (problems are the same) or "False" (problems are different). Do not respond with anything else.
'''.strip(),
        },
    ]
    assert (
        prompt.fill(
            {
                'problem1': "1 + 3",
                "problem2": "what's 3 plus 1?",
            }
        )
        == expected_prompt
    )


def test_generic_lean4_prompt():
    prompt = get_prompt('generic/lean4', 'deepseek-prover')

    expected_prompt = """
    <｜begin▁of▁sentence｜>
    Complete the following Lean 4 code:
    ```lean4
    import Mathlib
    import Aesop

    set_option maxHeartbeats 0

    open BigOperators Real Nat Topology Rat

    /-- The volume of a cone is given by the formula $V = \frac{1}{3}Bh$, where $B$ is the area of the base and $h$ is the height. The area of the base of a cone is 30 square units, and its height is 6.5 units. What is the number of cubic units in its volume? Show that it is 65.-/

    theorem mathd_algebra_478 (b h v : ℝ) (h₀ : 0 < b ∧ 0 < h ∧ 0 < v) (h₁ : v = 1 / 3 * (b * h))
        (h₂ : b = 30) (h₃ : h = 13 / 2) : v = 65 := by

"""
    assert prompt.fill({"header": "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n", "informal_prefix": "/-- The volume of a cone is given by the formula $V = \\frac{1}{3}Bh$, where $B$ is the area of the base and $h$ is the height. The area of the base of a cone is 30 square units, and its height is 6.5 units. What is the number of cubic units in its volume? Show that it is 65.-/\n", "formal_statement": "theorem mathd_algebra_478 (b h v : \u211d) (h\u2080 : 0 < b \u2227 0 < h \u2227 0 < v) (h\u2081 : v = 1 / 3 * (b * h))\n    (h\u2082 : b = 30) (h\u2083 : h = 13 / 2) : v = 65 := by\n"}) == expected_prompt