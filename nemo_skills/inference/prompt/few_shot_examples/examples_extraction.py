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
        \[ x = \begin{cases} 0 & \\text{if } a = 0 \\ \frac{3}{4}a & \text{if } a > 0 \\ \text{no solution} & \text{if } a < 0 \end{cases} \]
""".strip(),
    },
    {
        "original_answer": "$5 \\le x \\le 10.$",
        "generation": """
        [5, 10]
""".strip(),
    },
    {
        "original_answer": "$x_1=\\frac{a-b}{c}$; $x_2=-\\frac{b}{c}$.",
        "generation": """
        (\\frac{a-b}{c}, -\\frac{b}{c})
""".strip(),
    },
    {
        "original_answer": "$x_1=a$; $x_2=b$.",
        "generation": """
        (a, b)
""".strip(),
    },
    {
        "original_answer": "$365\\frac{5}{8}.$",
        "generation": """
        365\\frac{5}{8}
""".strip(),
    },
    {
        "original_answer": "$(a+c)^2-b^2$; $139\\frac{91}{225}$.",
        "generation": """
        ((a+c)^2-b^2, $139\\frac{91}{225})
""".strip(),
    },
    {
        "original_answer": "$x_1=\\frac{n+1}{n-1}$; $x_2=-1$.",
        "generation": """
        (\\frac{n+1}{n-1}, -1)
""".strip(),
    },
    {
        "original_answer": "(1) $x=3,\\ y=2$; (2) $x=2,\\ y=3$.",
        "generation": """
        ((3, 2), (2, 3))
""".strip(),
    },
    {
        "original_answer": "3487 roub. 50 kop.",
        "generation": """
        3487.50
""".strip(),
    },
    {
        "original_answer": "The weight of the cut-off piece is (approximately) equal to 116.2 kg.",
        "generation": """
        116.2
""".strip(),
    },
    {
        "original_answer": "2 liters.",
        "generation": """
        2
""".strip(),
    },
    {
        "original_answer": "$V=\\frac{1}{2}S\\cdot l\\cos(\\alpha -\\beta )$.",
        "generation": """
        \\frac{1}{2}S\\cdot l\\cos(\\alpha -\\beta)
""".strip(),
    },
    {
        "original_answer": "$a_n=p+q-n$.",
        "generation": """
        p+q-n
""".strip(),
    },
    {
        "original_answer": "17 kg of copper, 7 kg of zinc.",
        "generation": """
        (17, 7)
""".strip(),
    },
    {
        "original_answer": "The arc smaller than $\frac{\pi }{2}$ is equal to $\arccos \frac{m-n}{m+n}$ $(m>n)$; the arc larger than $\frac{\pi }{2}$ is equal to $\pi =\arccos \frac{m-n}{m+n}=\arccos \frac{n-m}{m+n}$.",
        "generation": r"""
        \[ \text{arc} = \begin{cases} \arccos \frac{m-n}{m+n} & \text{if arc is smaller than } \frac{\pi}{2} \\ \arccos \frac{n-m}{m+n} & \text{if arc is larger than } \frac{\pi}{2} \end{cases} \]
""".strip(),
    },
    {
        "original_answer": "$S=2\pi b^2\tan\alpha (\cos\alpha +1-2\cos^2 \alpha ) =4\pi b^2\tan\alpha \sin\frac{\alpha }{2}\sin\frac{3\alpha }{3}$.",
        "generation": """
        4\\pi b^2\\tan\\alpha \\sin\\frac{\\alpha }{2}\\sin\\frac{3\\alpha }{3}
""".strip(),
    },
    {
        "original_answer": "$x=-\\frac{\\pi }{4}+\\pi n$.",
        "generation": """
        -\\frac{\\pi }{4}+\\pi n
""".strip(),
    },
    {
        "original_answer": "$bx^2-2a\\sqrt {ax}+a^2=0$.",
        "generation": """
        bx^2-2a\\sqrt {ax}+a^2=0
""".strip(),
    },
]

from_solution_extraction = [
    {
        "question": """
        Solve the equation $1+\\sin x+\\cos x+\\sin 2x+\\cos 2x=0.$
        """.strip(),
        "solution": """
    The equation can be written in the following form: $$(\sin x+\cos x)^2+(\sin
    x+\cos x)+(\cos^2 x-\sin^2 x)=0,$$ that is $$(\sin x+\cos x)(1+2\cos x)=0.$$
    Equating each of the expressions in the brackets to zero, we find the roots.
    {\it Answer:} $x_1=-\frac{\pi }{4}+k\pi,\ x_2=\pm \frac{2\pi }{3}+2k\pi $.
    """,
        "generation": """
    (\frac{\pi }{4}+k\pi, \frac{2\pi }{3}+2k\pi)
""".strip(),
    },
    {
        "question": """
        Given an arithmetic progression $a_1,\\ldots,a_n,a_{n+1},\\ldots $ prove that the equalities $$a_1^2-C_n^1 a_2^2+\\ldots +(-1)^n C_n^n a_{n+1}^2=0$$ hold for $n\\ge 3$.
        """.strip(),
        "solution": """
        We carry out the proof by induction. For $n=3$ it readily follows that
        $$a_1^2-3(a_1+d)^2+3(a_1+2d)^2-(a_1+3d)^2=0.$$ Suppose we have already
        established that for a certain $n$ and any arithmetic progression
        $x_1,x_2,\ldots,x_{n+1}$ the identity $$x_1^2-C_n^1 x_2^2+\ldots +(-1)^n C_n^n
        x_{n+1}^2=0$$ holds. Then passing to $n+1$ as in the preceding problem we obtain
        $$a_1^2-C_{n+1}^1 a_2^2+C_{n+1}^2 a_3^2+\ldots +(-1)^n C_{n+1}^n a_{n+1}^2
        +(-1)^{n+1}C_{n+1}^{n+1}a_{n+2}^2$$ $$=[a_1^2-C_n^1 a_2^2+\ldots +(-1)^n C_n^n
        a_{n+1}^2] -[a_2^2-C_n^1 a_3^2+\ldots +(-1)^n C_n^n a_{n+2}^2]=0,$$ and thus the
        required formula has been proved. It should be noted that for an arithmetic
        progression $a_1,a_2,\ldots,a_n,a_{n+1}$ the more general formula $$a_1^k-C_n^1
        a_2^k+C_n^2 a_3^k-\ldots +(-1)^{n-1}C_n^{n-1}a_n^k +(-1)^n C_n^n a_{n+1}^k=0$$
        holds where $k\ge 1$ is an integer.
        """,
        "generation": """
        prove_question
""".strip(),
    },
    {
        "question": """
        Solve the equation $\\frac{1-\\tan x}{1+\\tan x}=1+\\sin 2x.$       
        """.strip(),
        "solution": """
        The equation makes no sense for $x=\frac{\pi }{2}+k\pi $ and for $x=-\frac{\pi
        }{4}+k\pi $. For all the other values of $x$ it is equivalent to the equation
        $$\frac{\cos x-\sin x}{\cos x+\sin x}=1+\sin 2x.$$ After simple transformations
        we obtain $$\sin x(3+\sin 2x+\cos 2x)=0.$$ It is obvious that the equation $\sin
        2x+\cos 2x+3=0$ has no solution, and therefore, the original equation is reduced
        to the equation $\sin x=0$. {\it Answer:} $x=k\pi $.
        """,
        "generation": """
        k\pi
""".strip(),
    },
    {
        "question": """
        In how many ways can a pack of 36 cards be split in two so that each portion contains two aces?
        """.strip(),
        "solution": """
        Any splitting of the pack indicated in the statement of the problem is
        equivalent to selecting 16 cards out of the 32 cards that are not aces and two
        aces out of the four aces. The first selection can be accomplished in
        $C_{32}^{16}$ ways, and the second in $C_4^2$ ways. Since every selection of the
        above 16 cards can be combined with any selection of two aces, the total number
        of ways in which the pack can be split is equal to $C_{32}^{16}C_4^2$.
        """,
        "generation": """
        C_{32}^{16}C_4^2
""".strip(),
    },
    {
        "question": """ 
        Solve the equation $2+\\cos x=2\\tan \\frac{x}{2}.$"
        """.strip(),
        "solution": """
        Write the equation in the following form: $$\cos^2 \frac{x}{2}-\sin^2
        \frac{x}{2} =2\left(\frac{\sin\frac{x}{2}}{\cos\frac{x}{2}}-1\right).$$ After
        some simple transformations it is reduced to the equation
        $$\left(\cos\frac{x}{2}-\sin\frac{x}{2}\right) \left(3\cos^2 \frac{x}{2}+2\sin^2
        \frac{x}{2}+\sin\frac{x}{2}\cos\frac{x}{2} \right)=0.$$ The equation $$3\cos^2
        \frac{x}{2}+2\sin^2 \frac{x}{2}+\sin\frac{x}{2}\cos\frac{x}{2}=0$$ is equivalent
        to the equation $$2\tan^2 \frac{x}{2}+\tan\displaystyle\frac{x}{2}+3=0$$ and has
        no real solutions. {\it Answer:} $x=\displaystyle\frac{\pi }{2}+2k\pi $.
        """,
        "generation": """
        \frac{\pi}{2} + 2k\pi
""".strip(),
    },
    {
        "question": """
        Solve the equation $\\sin\\left(\\frac{\\pi }{10}+\\frac{3x}{2}\\right)=2\\sin\\left(\\frac{3\\pi }{10} -\\frac{x}{2}\\right).$
        """.strip(),
        "solution": """
        Put $\frac{3\pi }{10}-\frac{x}{2}=y$, then $\frac{\pi }{10}+\frac{3x}{2}=\pi
        -3\left(\frac{3\pi }{10}-\frac{x}{2}\right) =\pi -3y$, and the equation takes
        the form $$\sin 3y=2\sin y.$$ With the aid of formula (7), page 73, the last
        equation can be transformed to the form $$\sin y(4\sin ^2 y-1)=0. \eqno(1)$$
        Equation (1) has the following solutions: $$y_1=k\pi , y_2=(-1)^k \frac{\pi
        }{6}+k\pi , y_3=(-1)^{k+1}\frac{\pi }{6}+k\pi.$$ Returning to the argument
        $x=\frac{3\pi }{5}-2y$ we finally obtain the solutions of the original equation:
        $$x_1=\frac{3\pi }{5}-2k\pi , x_2=\frac{3\pi }{5}+(-1)^{k+1}\frac{\pi }{3}-k\pi
        , x_3=\frac{3\pi }{5}+(-1)^k \frac{\pi }{3}-k\pi .$$
        """,
        "generation": """
        (\frac{3\pi }{5}-2k\pi , \frac{3\pi }{5}+(-1)^{k+1}\frac{\pi }{3}-k\pi , \frac{3\pi }{5}+(-1)^k \frac{\pi }{3}-k\pi)
        
""".strip(),
    },
    {
        "question": """
        Two persons deposited equal sums of money in a savings bank. One of them withdrew his money after $m$ months and received $p$ roubles, and the other withdrew the money after $n$ months and received $q$ roubles. How much money did either person deposit and what interest does the savings bank pay?
        """.strip(),
        "solution": """
        If $x$ is the original sum of money each person deposited and $y$ is the
        interest paid by the savings bank, then $$x+x\frac{y}{100}\cdot \frac{m}{12}=p,
        x+x\frac{y}{100}\cdot \frac{n}{12}=q.$$ Multiplying the first equation by $n$
        and the second by $m$, and subtracting the latter equation from the former, we
        find $$x=\frac{pn-qm}{n-m}.$$ Now taking the original system and subtracting the
        second equation from the first one we get $$\frac{xy}{1200}(m-n)=p-q$$ whence we
        obtain $$y=\frac{1200(p-q)}{qm-pn}\%.$$
        """,
        "generation": """
        \(\left( \frac{pn-qm}{n-m}, \frac{1200(p-q)}{qm-pn} \right)\)
""".strip(),
    },
    {
        "question": """
        A right triangle with legs $a_1$ and $b_1$ is cut off from a quadrilateral with sides $a$ and $b$. How must the quadrilateral of maximum area with sides parallel to those of the initial quadrilateral be cut off from the remaining part of the quadrilateral?
        """.strip(),
        "solution": """
        Let $x$ be the distance between the bank the travelers started from and the
        Let a right triangle with vertex $C$ and legs $a_1$ and $b_1$ be cut off from a
        rectangle $ABCD$ with sides $a$ and $b$. Consider the pentagon $ABEFD$ thus
        obtained (Fig. 137). It is clear that one of the vertices (say $C_1$) of the
        sought-for rectangle $AB_1C_1D_1$ must lie on the line segment $EF$. The problem
        is thus reduced to finding the position of this vertex.  To find the point $C_1$
        extend the sides $AB$ and $AD$ of the rectangle to intersect the extension of
        the line segment $EF$. This results in a triangle $AMN$. Let $$AM=m, AN=n
        \mbox{and} B_1C_1=AD_1=x.$$ The similarity of the triangles $AMN$ and $D_1C_1N$
        implies that $$\frac{C_1D_1}{m}=\frac{n-x}{n},$$ whence we find
        $$C_1D_1=\frac{m}{n}(n-x).$$ Hence, for the area $S$ of the rectangle
        $AB_1C_1D_1$ which is equal to $AD_1\cdot C_1D_1$ we get the expression
        $$S=\frac{m}{n}(n-x)x.$$ Transforming this expression to the form
        $$S=\frac{m}{n}\left[\frac{n^2}{4}-\left(\frac{n}{2}-x\right)^2\right],
        \eqno(1)$$ we conclude that the greatest value of $S$ is attained when
        $\frac{n}{2}-x=0$, i.e. for $x=\frac{n}{2}$. Let $C_0$ be the position of the
        vertex $C_1$ corresponding to $x=\frac{n}{2}$. Noting that expression (1) for
        $S$ decreases when $\left|\frac{n}{2}-x\right|$ increases, i.e. when the point
        $C_1$ moves from the point $C_0$ to the vertex $M$ or $F$, we find that there
        are three possible cases here, namely: (1) The point $C_0$ lies on the line
        segment $EF$; then the vertex $C_1$ of the required rectangle coincides with
        $C_0$. (2) The point $C_0$ lies on the line segment $ME$; then $C_1$ must
        coincide with $E$. (3) The point $C_0$ lies on the line segment $FN$; then $C_1$
        must coincide with $F$. We now must establish a criterion for distinguishing
        between these cases with the aid of the magnitudes of the quantities $a$, $a_1$,
        $b$ and $b_1$ given in the formulation of the problem. Let us first find the
        quantity $n$. The similarity of the triangles $ECF$ and $NDF$ implies that
        $$\frac{n-b}{a-a_1}=\frac{b_1}{a_1}$$ whence we find
        $$n=b+\frac{b_1}{a_1}(a-a_1). \eqno(2)$$ Now note that the point $C_0$ is within
        the line segment $EF$ if the inequalities $b-b_1<x<b$ are fulfilled.
        Substituting $x=\frac{n}{2}$ with the known value of $n$ into the above we
        obtain $$b-b_1<\frac{b}{2}+\displaystyle\frac{b_1}{2a_1}(a-a_1)<b.$$ The latter
        inequalities are readily transformed to the form
        $$-1<\displaystyle\frac{a}{a_1}-\displaystyle\frac{b}{b_1}<1. \eqno(3)$$ If the
        inequality $-1<\displaystyle\frac{a}{a_1}-\displaystyle\frac{b}{b_1}$ is
        violated, the point $C_0$ falls on the line segment $ME$, and if the inequality
        $\displaystyle\frac{a}{a_1}-\displaystyle\frac{b}{b_1}<1$ does not hold, $C_0$
        falls on $FN$. Thus, we arrive at the following final results: if for given $a$,
        $b$, $a_1$ and $b_1$ both inequalities (3) are fulfilled, then the vertex $C_1$
        of the rectangle of the greatest area lies within the line segment $EF$, and the
        side $x$ of this rectangle is computed by the formula
        $$x=\displaystyle\frac{b}{2}+\displaystyle\frac{b}{2a_1}(a-a_1).$$ If the left
        inequality in (3) does not hold true, the vertex $C_1$ coincides with the point
        $E$, and if the right inequality is not fulfilled, then $C_1$ coincides with
        $F$.
        """,
        "generation": """
        \[
        \begin{cases}
        \frac{b}{2} + \frac{b_1}{2a_1}(a - a_1) & \text{if} -1 < \frac{a}{a_1} - \frac{b}{b_1} < 1 \\
        E & \text{if } -1 \geq \frac{a}{a_1} - \frac{b}{b_1} \\
        F & \text{if } \frac{a}{a_1} - \frac{b}{b_1} \geq 1
        \end{cases}
        \]
""".strip(),
    },
    {
        "question": """
        Solve the system of equations $$\\left\\{\\begin{array}{lll} \\tan x+\\tan y=1,\\medskip \\\\ \\cos x\\cos y=\\frac{1}{\\sqrt 2}. \\end{array}\\right.$$
        """.strip(),
        "solution": """
        The first equation can be written in the form $$\frac{\sin (x+y)}{\cos x\cos
        y}=1,$$ whence, by virtue of the second equation, we obtain $$\sin (x+y)=\cos
        x\cos y=\frac{\sqrt 2}{2}.$$ Hence, either $$x+y=\frac{\pi }{4}+2k\pi \eqno(1)$$
        or $$x+y=-\frac{\pi }{4}+(2k+1)\pi . \eqno(2)$$ The second equation of the
        original system can be transformed to the form $$\cos (x+y)+\cos (x-y)=\sqrt
        2.$$ It follows that $$\cos (x-y)=\sqrt 2-\cos (x+y). \eqno(3)$$ If (1) holds,
        then $\cos (x+y)=\frac{\sqrt 2}{2}$, and from (3) we find $$\cos
        (x-y)=\frac{\sqrt 2}{2}, x-y=\pm\frac{\pi }{4}+2l\pi .$$ If (2) holds, then
        $\cos (x+y)=-\frac{\sqrt 2}{2}$ and $\cos (x-y)=\frac{3\sqrt 2}{2}$ which is
        impossible. Thus we have the system of equations $$\left\{\begin{array}{lll}
        x+y=\frac{\pi }{4}+2k\pi ,\medskip \\ x-y=\pm \frac{\pi }{4}+2l\pi
        \end{array}\right. \eqno(4)$$ for finding $x$ and $y$. According to the choice
        of the sign in the second equation of system (4), we obtain the solutions
        $$x_1=\frac{\pi }{4}+(k+l)\pi , y_1=(k-l)\pi $$ and $$x_2=(k+l)\pi ,
        y_2=\frac{\pi }{4}+(k-l)\pi .$$
        """,
        "generation": """
        (\frac{\pi }{4}+(k+l)\pi, (k-l)\pi), ((k+l)\pi, \frac{\pi }{4}+(k-l)\pi)
""".strip(),
    },
]


math_soution_extraction = [
    {
        "question": """
        What is the largest number, all of whose digits are 3 or 2, and whose digits add up to $11$?
        """.strip(),
        "solution": """
        To make a number as large as possible, we want as many digits as possible, so we want the digits to be as small as possible. To have the most number of digits, we use 4 twos and 1 three to make $4 \\cdot 2 +3 =11$. We want to arrange them in decreasing order because we want the digits to the left to be as large as possible. Therefore, we have the number $32222$.
        """.strip(),
        "original_answer": """
        32222
    """.strip(),
    },
    {
        "question": """
        What is $a-2b$, where $a=4-2i$ and $b=3+2i$?
        """.strip(),
        "solution": """
        Substituting in, we have $4-2i-2(3+2i)$. Expanding the last part, we have $4-2i-6-4i$; adding, we have $-2-6i$.
        """.strip(),
        "original_answer": """
        (2, 6)
    """.strip(),
    },
    {
        "question": """
            There are 3 math teams in the area, with 5, 7, and 8 students respectively. Each team has two co-captains. If I randomly select a team, and then randomly select two members of that team to give a copy of $\\emph{Introduction to Geometry}$, what is the probability that both of the people who receive books are co-captains?
        """.strip(),
        "solution": """
        There's a $\\dfrac{1}{3}$ chance that I will select each team. Once I have selected a team, let $n$ be the number of students on that team. There are $\\dbinom{n}{2}$ ways to choose a pair of those students to give books to, but only one of those pairs will be the two co-captains, which means that once I have selected that team, the probability that I give books to the co-captains is $$\\dfrac{1}{\\dfrac{n(n-1)}{2}}=\\dfrac{2}{n(n-1)}.$$Since the teams have $5,$ $7,$ and $8$ students, this means that the total probability is $$\\dfrac{1}{3}\\left(\\dfrac{2}{5(5-1)}+\\dfrac{2}{7(7-1)}+\\dfrac{2}{8(8-1)}\\right)$$which after a bit of arithmetic simplifies to $\\dfrac{11}{180}$.
        """.strip(),
        "original_answer": """
        \\dfrac{11}{180}
        """.strip(),
    },
    {
        "question": """
        Let $(a,b,c,d)$ be a solution to the system\\begin{align*}a+b&=15,\\\\ab+c+d&=78,\\\\ad+bc&=160,\\\\cd&=96.\\end{align*}Find the greatest possible value of $a^2+b^2+c^2+d^2$.\n
        """.strip(),
        "solution": """
        Note that when multiplying quadratics, terms add up similar to the equations of a system, so let\\begin{align*} p(x) &= (x^2 + ax + c)(x^2 + bx + d) \\\\ &= x^4 + (a+b)x^3 + (ab+c+d)x^2 + (ad+bc)x + cd \\\\ &= x^4 + 15x^3 + 78x^2 + 160x + 96 \\end{align*}Factoring $p(x)$ with the Rational Root Theorem results in $(x+4)(x+4)(x+1)(x+6)$. By the Fundamental Theorem of Algebra, we know that $x+4, x+4, x+1, x+6$ are all the linear factors of the polynomial, so the quadratic factors can only be multiplied from these linear factors.\nThere are only two possible distinct groupings (not counting rearrangements) -- $(x^2 + 8x + 16)(x^2 + 7x + 6)$ and $(x^2 + 5x + 4)(x^2 + 10x + 24)$. In the first case, $a^2 + b^2 + c^2 + d^2 = 405$, and in the second case, $a^2 + b^2 + c^2 + d^2 = 717$. The largest of the two options is $717$.
        """.strip(),
        "original_answer": """
        717
        """.strip(),
    },
    {
        "question": """
        In triangle $ABC$, altitudes $AD$, $BE$, and $CF$ intersect at the orthocenter $H$.  If $\\angle ABC = 49^\\circ$ and $\\angle ACB = 12^\\circ$, then find the measure of $\\angle BHC$, in degrees.

        """.strip(),
        "solution": """
        Note that triangle $ABC$ is obtuse, so $H$ lies outside triangle $ABC$.\n\n[asy]\nunitsize(1 cm);\n\npair A, B, C, D, E, F, H;\n\nB = (0,0);\nC = (4,0);\nA = extension(B, B + dir(49), C, C + dir(180 - 12));\nD = (A + reflect(B,C)*(A))/2;\nE = (B + reflect(C,A)*(B))/2;\nF = (C + reflect(A,B)*(C))/2;\nH = extension(B,E,C,F);\n\ndraw(B--H--C--cycle);\ndraw(H--D);\ndraw(B--F);\ndraw(C--E);\n\nlabel(\"$A$\", A, SE);\nlabel(\"$B$\", B, SW);\nlabel(\"$C$\", C, SE);\nlabel(\"$D$\", D, S);\nlabel(\"$E$\", E, W);\nlabel(\"$F$\", F, NE);\nlabel(\"$H$\", H, N);\n[/asy]\n\nSince triangle $BEC$ is right, $\\angle CBE = 90^\\circ - \\angle BCE = 90^\\circ - 12^\\circ = 78^\\circ$.  Since triangle $BFC$ is right, $\\angle BCF = 90^\\circ - \\angle CBF = 90^\\circ - 49^\\circ = 41^\\circ$.  Therefore, $\\angle BHC = 180^\\circ - \\angle CBH - \\angle BCH = 180^\\circ - 78^\\circ - 41^\\circ = 61^\\circ$.
        """.strip(),
        "original_answer": """
        61
        """.strip(),
    },
    {
        "question": """
        The complement of an angle is $5^{\\circ}$ more than four times the angle. What is the number of degrees in the measure of the angle?
        """.strip(),
        "solution": """
        Let the measure of the angle be $x$, so $5^\\circ$ more than four times the angle is $4x + 5^\\circ$.  Since these two measures are complementary, we have $x + (4x+5^\\circ) = 90^\\circ$.  Simplifying the left side gives $5x+5^\\circ = 90^\\circ$, so $5x = 85^\\circ$ and $x = 17^\\circ$.
        """.strip(),
        "original_answer": """
        17
        """.strip(),
    },
    {
        "question": """
        For all real numbers $x$ except $x=0$ and $x=1$ the function $f(x)$ is defined by\n\\[f \\left( \\frac{x}{x - 1} \\right) = \\frac{1}{x}.\\]Suppose $0\\leq t\\leq \\frac{\\pi}{2}$. What is the value of $f(\\sec^2t)$?
        """.strip(),
        "solution": """
        First, we must solve\n\\[\\frac{x}{x - 1} = \\sec^2 t.\\]Solving for $x,$ we find $x = \\frac{\\sec^2 t}{\\sec^2 t - 1}.$  Then\n\\[f(\\sec^2 t) = \\frac{1}{x} = \\frac{\\sec^2 t - 1}{\\sec^2 t} = 1 - \\cos^2 t = \\sin^2 t.\\]
        """.strip(),
        "original_answer": """
        \\sin^2 t
        """.strip(),
    },
]

examples_map = {
    "antonov_tuple_extraction": tuple_extraction,
    "lidsky_solution_extraction": from_solution_extraction,
    "math_soution_extraction": math_soution_extraction,
}
