import sympy as sp

def simplify(expr):
    # 替换 '^' 为 '**' 以符合 SymPy 的处理规则
    expr = expr.replace('^', '**')

    # 使用 SymPy 解析表达式
    expression = sp.nsimplify(sp.sympify(expr))

    # 简化表达式
    simplified_expr = '(' + str(expression) + ')'
    
    return simplified_expr