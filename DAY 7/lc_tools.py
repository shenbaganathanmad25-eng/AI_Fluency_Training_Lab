"""Day 7, Part B: three custom tools, each built a different way."""
import ast
import operator
from typing import Literal
from pydantic import BaseModel, Field
from langchain.tools import tool

COURSE_FEES = {"CS101": 12000, "AI202": 18000, "DS303": 15000}
TIMETABLE = {
    "monday":    "09:00 CS101 lecture, 14:00 AI202 lab",
    "tuesday":   "10:00 DS303 lecture",
    "wednesday": "09:00 AI202 lecture, 15:00 CS101 tutorial",
    "thursday":  "11:00 DS303 lab",
    "friday":    "09:00 AI Fluency Training review",
}

# ---------- Tool 1: plain @tool, schema from type hints and docstring ----------
@tool
def get_course_fee(course_code: str) -> str:
    """Get the fee in rupees for ONE course code, for example CS101.
    Valid codes: CS101, AI202, DS303. Returns the number only."""
    fee = COURSE_FEES.get(course_code.strip().upper())
    return str(fee) if fee is not None else (
        f"Unknown course code: {course_code}. Valid: {', '.join(COURSE_FEES)}")

# ---------- Tool 2: @tool(parse_docstring=True), argument described in Args ----------
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

@tool(parse_docstring=True)
def calculator(expression: str) -> str:
    """Evaluate one arithmetic expression and return the result.

    Args:
        expression: Digits, brackets and + - * / only, for example (12000 + 18000) * 0.9
    """
    try:
        return str(_evaluate(ast.parse(expression, mode="eval").body))
    except Exception as error:
        return f"Calculator error: {error}. Use only numbers and + - * / ( )."

# ---------- Tool 3: @tool(args_schema=...), a Pydantic schema with an enum ----------
class TimetableArgs(BaseModel):
    day: Literal["monday", "tuesday", "wednesday", "thursday", "friday"] = Field(
        description="Day of the week, in lower case")

@tool(args_schema=TimetableArgs)
def get_timetable(day: str) -> str:
    """Get the class timetable for one weekday. Weekends have no classes."""
    return TIMETABLE.get(day.lower(), f"No timetable for {day}.")

TOOLS = [get_course_fee, calculator, get_timetable]