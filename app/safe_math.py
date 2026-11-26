"""
A safe arithmetic evaluator to replace eval().

Python's ast module turns text into a tree of parts (numbers, operators,
brackets). We walk that tree and allow only the parts we list. Anything else,
such as a function call or an attribute access, is rejected before it runs.
"""

import ast
import operator

ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def safe_eval(expression: str) -> float:
    tree = ast.parse(expression.strip(), mode="eval")
    return _evaluate(tree.body)


def _evaluate(node: ast.AST) -> float:
    # A plain number, such as 12 or 2.5 (bools are excluded on purpose).
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    # A binary operation, such as left + right.
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
        return ALLOWED_OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    # A negative number, such as -5.
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_evaluate(node.operand)
    # Anything else (function calls, names, attributes...) is refused.
    raise ValueError("unsupported expression")


if __name__ == "__main__":
    for expr in ["12 * (3 + 4)", "100 / 4", "-5 + 2", "__import__('os')"]:
        try:
            print(expr, "=", safe_eval(expr))
        except (ValueError, SyntaxError, ZeroDivisionError) as exc:
            print(expr, "-> refused:", exc)
