chapter_1_ariphmetic_calculations = [
    {
        "question": "$\\frac{\\left(152\\frac{3}{4}-148\\frac{3}{8}\\right)\\cdot 0.3}{0.2}$",
        "generation": """
        First, convert the mixed numbers to improper fractions:
        \[ 152\frac{3}{4} = \frac{152 \times 4 + 3}{4} = \frac{611}{4} \]
        \[ 148\frac{3}{8} = \frac{148 \times 8 + 3}{8} = \frac{1187}{8} \]

        Then convert all the decimals into fractions: \\
        \[ 0.3 = \frac{3}{10} \] \\
        \[ 0.2 = \frac{2}{10} = \frac{1}{5} \]

        Next, subtract the two improper fractions:
        \[ \frac{611}{4} - \frac{1187}{8} = \frac{611 \times 2}{8} - \frac{1187}{8} = \frac{1222}{8} - \frac{1187}{8} = \frac{35}{8} \]

        Multiply the result by \(\frac{3}{10}\):
        \[ \left(\frac{35}{8}\right) \cdot \frac{3}{10} = \frac{35 \times 3}{80} = \frac{105}{80} = \frac{21}{16} \]

        Finally, divide by \(\frac{1}{5}\):
        \[ \frac{\frac{21}{16}}{\frac{1}{5}} = \frac{21}{16} \cdot 5 = \frac{105}{16} \]

        Thus, the answer is \boxed{6.5625}.
""".strip(),
    },
    {
        "question": "$\\frac{\\left(68\\frac{7}{30}-66\\frac{5}{18}\\right):6\\frac{1}{9} +\\left(\\frac{7}{40}+\\frac{3}{32}\\right)\\cdot 4.5}{0.04}$",
        "generation": """
        First, convert all mixed numbers to improper fractions: \\
        \[ 68\frac{7}{30} = \frac{68 \times 30 + 7}{30} = \frac{2040 + 7}{30} = \frac{2047}{30} \] \\
        \[ 66\frac{5}{18} = \frac{66 \times 18 + 5}{18} = \frac{1188 + 5}{18} = \frac{1193}{18} \] \\
        \[ 6\frac{1}{9} = \frac{6 \times 9 + 1}{9} = \frac{54 + 1}{9} = \frac{55}{9} \]

        Then convert all the decimals into fractions: \\
        \[ 4.5 = \frac{45}{10} = \frac{9}{2} \] \\
        \[ 0.04 = \frac{4}{100} = \frac{1}{25} \]

        Subtract the two improper fractions: \\
        \[ \frac{2047}{30} - \frac{1193}{18} = \frac{2047 \times 6}{180} - \frac{1193 \times 10}{180} = \frac{12282 - 11930}{180} = \frac{352}{180} = \frac{176}{90} = \frac{88}{45} \]

        Divide the result by \(6\frac{1}{9}\): \\
        \[ \frac{88}{45} : \frac{55}{9} = \frac{88}{45} \times \frac{9}{55} = \frac{792}{2475} = \frac{24}{75} = \frac{8}{25} \]

        Add the fractions: \\
        \[ \frac{7}{40} + \frac{3}{32} = \frac{7 \times 8}{320} + \frac{3 \times 10}{320} = \frac{56 + 30}{320} = \frac{86}{320} = \frac{43}{160} \]

        Multiply by \(\frac{9}{2}\): \\
        \[ \frac{43}{160} \cdot \frac{9}{2} = \frac{43 \times 9}{160 \times 2} = \frac{387}{320} \]

        Sum of results: \\
        \[ \frac{8}{25} + \frac{387}{320} = \frac{8 \times 64 + 387 \times 5}{1600} = \frac{512 + 1935}{1600} = \frac{2447}{1600} \]

        Divide by \(\frac{1}{25}\): \\
        \[ \frac{2447}{1600} : \frac{1}{25} = \frac{2447}{1600} \times 25 = \frac{61175}{1600} = 38 \frac{15}{64} \]

        Thus, the answer is \[ \boxed{38 \frac{15}{64}} \].
""".strip(),
    },
    {
        "question": "$26:\\left[\\frac{3:(0.2-0.1)}{2.5\\cdot (0.8+1.2)} +\\frac{(34.06-33.81)\\cdot 4}{6.84:(28.57-25.15)}\\right] +\\frac{2}{3}:\\frac{4}{21}$",
        "generation": """
        First, convert all decimals to fractions: \\
        \[ 0.2 = \frac{1}{5}, \quad 0.1 = \frac{1}{10}, \quad 0.8 = \frac{4}{5}, \quad 1.2 = \frac{6}{5}, \quad 34.06 = \frac{3406}{100}, \quad 33.81 = \frac{3381}{100} \]
        \[ 6.84 = \frac{684}{100}, \quad 28.57 = \frac{2857}{100}, \quad 25.15 = \frac{2515}{100} \]

        Calculate the first part of the bracket: \\
        \[ \frac{3 : \left(\frac{1}{5} - \frac{1}{10}\right)}{\frac{5}{2} \cdot \left(\frac{4}{5} + \frac{6}{5}\right)} = \frac{3 : \frac{1}{10}}{\frac{5}{2} \cdot 2} = \frac{30}{5} = 6 \]

        Calculate the second part of the bracket: \\
        \[ \frac{\left(\frac{3406}{100} - \frac{3381}{100}\right) \cdot 4}{\frac{684}{100} : \left(\frac{2857}{100} - \frac{2515}{100}\right)} = \frac{\frac{25}{100} \cdot 4}{\frac{684}{100} : \frac{342}{100}} = \frac{1}{2} \]
        Thus the bracket simplifies to: \\
        \[ 6 + \frac{1}{2} = 6.5 = \frac{13}{2} \]

        Compute the main expression: \\
        \[ 26 : \frac{13}{2} + \frac{2}{3} : \frac{4}{21} = 26 \cdot \frac{2}{13} + \frac{2}{3} \cdot \frac{21}{4} = 4 + \frac{42}{12} = 4 + \frac{7}{2} = 4 + 3\frac{1}{2} = 7\frac{1}{2} \]

        Thus, the answer is \[ \boxed{7\frac{1}{2}} \].
        \hline
""".strip(),
    },
    {
        "question": "$1.7:\\frac{\\left(4.5\\cdot 1\\frac{2}{3}+3.75\\right)\\cdot \\frac{7}{135}} {\\frac{5}{9}}-\\left(0.5+\\frac{1}{3}-\\frac{5}{12}\\right)$",
        "generation": """
        First, convert all decimals and mixed numbers to fractions: \\
        \[ 1.7 = \frac{17}{10}, \quad 4.5 = \frac{9}{2}, \quad 1\frac{2}{3} = \frac{5}{3}, \quad 3.75 = \frac{15}{4}, \quad 0.5 = \frac{1}{2} \]

        Compute the inner expressions and operations: \\
        \[ \frac{9}{2} \cdot \frac{5}{3} + \frac{15}{4} = \frac{45}{6} + \frac{15}{4} = \frac{30}{4} + \frac{15}{4} = \frac{45}{4} \]

        Multiply by \(\frac{7}{135}\) and divide by \(\frac{5}{9}\): \\
        \[ \left(\frac{45}{4} \cdot \frac{7}{135}\right) \cdot \frac{9}{5} = \frac{315}{540} \cdot \frac{9}{5} = \frac{7}{12} \cdot \frac{9}{5} = \frac{63}{60} = \frac{21}{20} \]

        Calculate the expression within the parentheses and subtract: \\
        \[ 0.5 + \frac{1}{3} - \frac{5}{12} = \frac{1}{2} + \frac{4}{12} - \frac{5}{12} = \frac{5}{12} \]

        Complete the final division and subtraction: \\
        \[ \frac{17}{10} : \frac{21}{20} - \frac{5}{12} = \frac{17}{10} \cdot \frac{20}{21} - \frac{5}{12} \]
        Find the common denominator for \(\frac{34}{21}\) and \(\frac{5}{12}\):
        \[ \frac{34 \times 12 - 5 \times 21}{252} = \frac{408 - 105}{252} = \frac{303}{252} = \frac{101}{84} = 1\frac{17}{84}\]

        Thus, the answer is \[ \boxed{1\frac{17}{84}} \].
""".strip(),

    },
    {
        "question": "$\\frac{\\left(1.75:\\frac{2}{3}-1.75\\cdot 1\\frac{1}{8}\\right) :\\frac{7}{12}}{\\left(\\frac{17}{80}-0.0325\\right):400} :(6.79:0.7+0.3)$",
        "generation": """
        First, convert all decimals and mixed numbers to fractions: \\
        \[ 1.75 = \frac{7}{4}, \quad 1\frac{1}{8} = \frac{9}{8}, \quad 0.0325 = \frac{13}{400}, \quad 6.79 = \frac{679}{100}, \quad 0.7 = \frac{7}{10} \]

        Calculate the numerator of the first fraction: \\
        \[ 1.75 : \frac{2}{3} - 1.75 \cdot 1\frac{1}{8} = \frac{7}{4} \cdot \frac{3}{2} - \frac{7}{4} \cdot \frac{9}{8} = \frac{21}{8} - \frac{63}{32} = \frac{21}{32}\]

        Now divide by \(\frac{7}{12}\):
        \[ \left(\frac{21}{32}\right) : \frac{7}{12} = \frac{21}{32} \cdot \frac{12}{7} = \frac{252}{224} = \frac{63}{56} = \frac{7}{8}\]

        Calculate the denominator of the first fraction: \\
        \[ \left(\frac{17}{80} - \frac{13}{400}\right) : 400 = \left(\frac{85}{400} - \frac{13}{400}\right) : 400 = \frac{72}{400} : 400 = \frac{72}{160000} = \frac{9}{20000} \]

        Compute the overall first fraction:
        \[ \frac{\frac{63}{56}}{\frac{9}{20000}} = \frac{63}{56} \cdot \frac{20000}{9} = \frac{1260000}{504} = \frac{2500}{1} \]

        Calculate the denominator of the overall expression: \\
        \[ 6.79 : 0.7 + 0.3 = \frac{679}{100} : \frac{7}{10} + \frac{3}{10} = \frac{679}{100} \cdot \frac{10}{7} + \frac{3}{10} = \frac{679}{70} + \frac{3}{10} = 10\]

        Now, divide the two results:
        \[ \frac{2500}{10} = 250 \]

        Thus, the answer is \[ \boxed{250} \].
""".strip(),
    },
]

chapter_2_Algebraic_Transformations = [
    {
        "question": "Simplify the following expressions: $(a^2-b^2-c^2+2bc):\\frac{a+b-c}{a+b+c}$ Evaluate the result at $a=8.6$; $b=\\sqrt 3$; $c=3\\frac{1}{3}$",
        "generation": """
        \[
        (a^2 - b^2 - c^2 + 2bc) : \frac{a+b-c}{a+b+c}
        \]

        Rewrite the division as a multiplication:
        \[
        (a^2 - b^2 - c^2 + 2bc) \cdot \frac{a+b+c}{a+b-c}
        \]

        Factorize \(a^2 - b^2 - c^2 + 2bc\):
        \[
        a^2 - b^2 - c^2 + 2bc = a^2 - (b^2 - 2bc + c^2) = a^2 - (b-c)^2 = (a - (b-c))(a + (b-c)) = (a - b + c)(a + b - c)
        \]

        Substitute back into the expression:
        \[
        (a - b + c)(a + b - c) \cdot \frac{a+b+c}{a+b-c}
        \]

        The \((a + b - c)\) terms cancel out:
        \[
        (a - b + c)(a + b + c) = (a + c)^2 - b^2
        \]

        Now, evaluate at \(a = 8.6\), \(b = \sqrt{3}\), \(c = 3\frac{1}{3}\):

        Convert the decimals into fractions:
        \[
        a = 8.6 = \frac{43}{5}, \quad c = 3\frac{1}{3} = \frac{10}{3}
        \]

        Substitute these values:
        \[
        \left(\frac{43}{5} + \frac{10}{3}\right)^2 - (\sqrt{3})^2
        \]

        Find a common denominator for the addition inside the parentheses:
        \[
        \frac{43}{5} + \frac{10}{3} = \frac{43 \cdot 3 + 10 \cdot 5}{5 \cdot 3} = \frac{129 + 50}{15} = \frac{179}{15}
        \]

        Square the fraction:
        \[
        \left(\frac{179}{15}\right)^2 = \frac{179^2}{15^2} = \frac{32041}{225}
        \]

        Subtract \((\sqrt{3})^2\):
        \[
        \frac{32041}{225} - 3 = \frac{32041}{225} - \frac{675}{225} = \frac{32041 - 675}{225} = \frac{31366}{225} = 139 \frac{91}{225}
        \]

        Thus, the evaluated result is:
        \[
        \boxed{139 \frac{91}{225}}
        \]
        
""".strip(),
    },
    {
        "question": "Simplify the following expressions: $\\left(\\sqrt {ab}-\\frac{ab}{a+\\sqrt {ab}}\\right): \\frac{\\sqrt [4]{ab}-\\sqrt {b}}{a-b}$",
        "generation": """
        \[
        \left(\sqrt{ab} - \frac{ab}{a + \sqrt{ab}}\right) : \frac{\sqrt[4]{ab} - \sqrt{b}}{a - b}
        \]

        \[
        \left(\sqrt{ab} - \frac{ab}{a + \sqrt{ab}}\right) : \frac{\sqrt[4]{ab} - \sqrt{b}}{a - b}
        \]

        First, rewrite the division as multiplication by the reciprocal:
        \[
        \left(\sqrt{ab} - \frac{ab}{a + \sqrt{ab}}\right) \cdot \frac{a - b}{\sqrt[4]{ab} - \sqrt{b}}
        \]

        Convert all roots to exponents:
        \[
        \left((ab)^{\frac{1}{2}} - \frac{ab}{a + (ab)^{\frac{1}{2}}}\right) \cdot \frac{a - b}{(ab)^{\frac{1}{4}} - b^{\frac{1}{2}}}
        \]

        Simplify the term \((ab)^{\frac{1}{2}} - \frac{ab}{a + (ab)^{\frac{1}{2}}}\):
        \[
        (ab)^{\frac{1}{2}} - \frac{ab}{a + (ab)^{\frac{1}{2}}} = \frac{(ab)^{\frac{1}{2}}(a + (ab)^{\frac{1}{2}}) - ab}{a + (ab)^{\frac{1}{2}}}
        \]
        \[
        = \frac{a(ab)^{\frac{1}{2}} + (ab) - ab}{a + (ab)^{\frac{1}{2}}} = \frac{a(ab)^{\frac{1}{2}}}{a + (ab)^{\frac{1}{2}}}
        \]

        Combine with the reciprocal:
        \[
        \frac{a(ab)^{\frac{1}{2}}}{a + (ab)^{\frac{1}{2}}} \cdot \frac{a - b}{(ab)^{\frac{1}{4}} - b^{\frac{1}{2}}}
        \]


        Continue simplifications
        \[
        \frac{a(ab)^{\frac{1}{2}}}{a^{\frac{1}{2}}(a^{\frac{1}{2}} + b^{\frac{1}{2}})} \cdot \frac{(a^{\frac{1}{2}} + b^{\frac{1}{2}})(a^{\frac{1}{4}}-b^{\frac{1}{4}})(a^{\frac{1}{4}}+b^{\frac{1}{4}})}{(a^{\frac{1}{4}} - b^{\frac{1}{4}})b^{\frac{1}{4}}}
        \]

        Canceling out some terms
        \[
        ab^{\frac{1}{4}}(a^{\frac{1}{4}}+b^{\frac{1}{4}})
        \]

        Thus,  result is:
        \[
        \boxed{ab^{\frac{1}{4}}(a^{\frac{1}{4}}+b^{\frac{1}{4}})}
        \]

""".strip(),
    },
    {
        "question": "Simplify the following expressions: $\\left[\\frac{a\\sqrt [3]{a}+\\sqrt [3]{a^2}}{a+\\sqrt [3]{a}}-\\sqrt [3]{x}\\right] \\left[\\left(\\sqrt [3]{a}-\\sqrt [3]{x}\\right)^2 +3\\left(\\sqrt [3]{a}+\\sqrt [3]{x}\\right)^2\\right]$",
        "generation": """
        Convert all roots to exponents:
        \[
        \frac{a^{\frac{4}{3}} + a^{\frac{2}{3}}}{a + a^{\frac{1}{3}}} - x^{\frac{1}{3}}
        \]

        \[
        \left[(a^{\frac{1}{3}} - x^{\frac{1}{3}})^2 + 3(a^{\frac{1}{3}} + x^{\frac{1}{3}})^2\right]
        \]

        Simplify the first bracket:
        \[
        \frac{a^{\frac{4}{3}} + a^{\frac{2}{3}}}{a + a^{\frac{1}{3}}} = \frac{a^{\frac{2}{3}}(a^{\frac{2}{3}} + 1)}{a^{\frac{1}{3}}(a^{\frac{2}{3}} + 1)} - x^{\frac{1}{3}}= a^{\frac{1}{3}} - x^{\frac{1}{3}}
        \]

        Simplify the second bracket:
        \[
        (a^{\frac{1}{3}} - x^{\frac{1}{3}})^2 + 3(a^{\frac{1}{3}} + x^{\frac{1}{3}})^2 = a^{\frac{2}{3}} - 2a^{\frac{1}{3}}x^{\frac{1}{3}} + x^{\frac{2}{3}} + 3(a^{\frac{2}{3}} + 2a^{\frac{1}{3}}x^{\frac{1}{3}} + x^{\frac{2}{3}}) = 4a^{\frac{2}{3}} + 4x^{\frac{2}{3}} + 4a^{\frac{1}{3}}x^{\frac{1}{3}}
        \]

        Combine both parts:
        \[
        (a^{\frac{1}{3}} - x^{\frac{1}{3}}) \times 4(a^{\frac{2}{3}} + x^{\frac{2}{3}} + a^{\frac{1}{3}}x^{\frac{1}{3}}) = 4(a - x)
        \]

        Thus, the result is:
        \[
        \boxed{4(a - x)}
        \]

""".strip(),
    },
]

chapter_7_algebraic_and_arithmetic_roblems = [
    {
        "question": "Find the weight of an artillery round, knowing that the charge weighs 0.8 kg, the weight of the projectile is equal to $\\frac{2}{3}$ of the total weight of the round, and the weight of the shell is $\\frac{1}{4}$ of the weight of the round.",
        "generation": """
        Let \( W \) represent the total weight of the artillery round. According to the problem, the weight of the projectile is \(\frac{2}{3}W\) and the weight of the shell is \(\frac{1}{4}W\). The charge weighs \(0.8 \text{ kg}\).

        First, express the total weight \( W \) as the sum of the weights of the projectile, shell, and charge:
        \[ W = \frac{2}{3}W + \frac{1}{4}W + 0.8 \]

        To find \( W \), solve the equation by first getting a common denominator for the fractions:
        \[ \frac{2}{3}W + \frac{1}{4}W = \frac{8}{12}W + \frac{3}{12}W = \frac{11}{12}W \]

        Substitute \(\frac{11}{12}W\) back into the total weight equation:
        \[ W = \frac{11}{12}W + 0.8 \]

        Isolate \( W \) on one side of the equation:
        \[ W - \frac{11}{12}W = 0.8 \]
        \[ \frac{1}{12}W = 0.8 \]

        Solve for \( W \):
        \[ W = 0.8 \times 12 \]
        \[ W = 9.6 \]

        Thus, the total weight of the artillery round is \boxed{9.6 \text{ kg}}.

""".strip(),
    },
    {
        "question": "A piece of a rail making 72\\% of the rail length is cut off. The remaining part weighs 45.2 kg. Determine the weight of the cut-off piece.",
        "generation": """
        Let \( L \) be the total weight of the entire rail. According to the problem, 72\% of the rail is cut off, and the remaining 28\% of the rail weighs 45.2 kg.

        First, express the weight of the remaining part of the rail in terms of \( L \):
        \[ 0.28L = 45.2 \]

        To find \( L \), solve the equation using a regular fraction for the division:
        \[ L = \frac{45.2}{0.28} = \frac{452}{2.8} = \frac{4520}{28} = \frac{1130}{7} \]

        Convert \( 72\% \) to a fraction:
        \[ 0.72 = \frac{72}{100} = \frac{18}{25} \]

        Now, find the weight of the cut-off piece, which is 72\% of the total weight:
        \[ 0.72L = \frac{18}{25} \times \frac{1130}{7} \]
        \[ 0.72L = \frac{20340}{175} \]

        To simplify \(\frac{20340}{175}\):
        \[ \frac{20340}{175} = \frac{1162}{10} = \frac{581}{5} = 116.2 \]

        Thus, the weight of the cut-off piece of the rail is \boxed{116.2} kg.
""".strip(),

    },
    {
        "question": "A student was asked to multiply 78 by a two-digit number in which the tens digit was three times as large as the units digit; by mistake, he interchanged the digits in the second factor and thus obtained a product smaller than the true product by 2808. What was the true product?",
        "generation": """
        Let the units digit be \( x \) and the tens digit be \( 3x \). The two-digit number is thus \( 30x + x = 31x \).

        When the digits are interchanged, the number becomes \( 13x \).

        The problem states that the error in the product, when the digits are interchanged, is 2808. We set up the equation:
        \[
        78 \times 31x - 78 \times 13x = 2808
        \]

        Simplify and solve for \( x \):
        \[
        78(31x - 13x) = 2808
        \]
        \[
        78 \times 18x = 2808
        \]
        \[
        18x = \frac{2808}{78}
        \]
        \[
        18x = 36
        \]
        \[
        x = 2
        \]

        The original number is \( 31x \):
        \[
        31 \times 2 = 62
        \]

        Calculate the true product:
        \[
        78 \times 62 = 4836
        \]

        The ansewr is  \boxed{4836}
""".strip(),
    },
]