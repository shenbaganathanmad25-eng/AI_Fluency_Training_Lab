"""Day 6: the same two tools, with stricter JSON Schemas."""
import ast
import operator

COURSE_FEES = {"CS101": 12000, "AI202": 18000, "DS303": 15000}
HOSTEL_CHARGE = {"odd": 4500, "even": 4500}

def get_course_fee(course_code: str, semester: str = "odd") -> str:
    """Return the fee for one course code."""
    fee = COURSE_FEES.get(course_code.strip().upper())
    if fee is None:
        return (f"Unknown course code: {course_code}. "
                f"Valid codes: {', '.join(COURSE_FEES)}")
    return str(fee)

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg}

def _evaluate(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_evaluate(node.operand))
    raise ValueError("Unsupported expression")

def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression."""
    try:
        return str(_evaluate(ast.parse(expression, mode="eval").body))
    except Exception as error:
        return f"Calculator error: {error}. Use only numbers and + - * / ( )."

TOOL_FUNCTIONS = {"get_course_fee": get_course_fee, "calculator": calculator}

TOOLS = [
    {"type": "function", "function": {
        "name": "get_course_fee",
        "description": ("Get the fee in rupees for ONE course code, for example CS101. "
                        "Returns the number only. Valid codes: CS101, AI202, DS303."),
        "parameters": {
            "type": "object",
            "properties": {
                "course_code": {"type": "string",
                                "description": "Course code such as CS101"},
                "semester": {"type": "string", "enum": ["odd", "even"],
                             "description": "Semester; defaults to odd"},
            },
            "required": ["course_code"],
            "additionalProperties": False,
        }}},
    {"type": "function", "function": {
        "name": "calculator",
        "description": ("Evaluate one arithmetic expression using + - * / and brackets, "
                        "for example (12000 + 18000) * 0.9. Returns the number only."),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string",
                               "description": "Arithmetic expression, digits and operators only"},
            },
            "required": ["expression"],
            "additionalProperties": False,
        }}},
]

# name -> the "parameters" block above, used by validate.py
SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}