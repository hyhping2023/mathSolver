import sympy as sp
from sympy.parsing.latex import parse_latex
import math

def math_calculator(expression:str):
    try:
        # result = str(sp.simplify(parse_latex(expression)))
        result = str(parse_latex(expression))
        return result
    except:
        return "Invalid expression"
    
if __name__ == "__main__":
    print(math_calculator("(40/100)*p"))
    print(math_calculator("\\%"))
