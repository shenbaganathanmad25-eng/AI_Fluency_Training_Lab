"""Day 7, Part B: look at what @tool generated - no model needed."""
import json
from langchain_core.utils.function_calling import convert_to_openai_tool
from lc_tools import TOOLS

for t in TOOLS:
    print("=" * 70)
    print("name        :", t.name)
    print("description :", t.description.splitlines()[0])
    print("args        :", json.dumps(t.args))

print("=" * 70)
print("\nThe JSON Schema actually sent to the model for get_timetable:\n")
print(json.dumps(convert_to_openai_tool(TOOLS[2]), indent=2))

print("\nCalling the tools directly:")
print("  get_course_fee('ai202')      ->", TOOLS[0].invoke({"course_code": "ai202"}))
print("  calculator('(12000+18000)*0.9') ->", TOOLS[1].invoke({"expression": "(12000+18000)*0.9"}))
print("  get_timetable('monday')      ->", TOOLS[2].invoke({"day": "monday"}))
try:
    TOOLS[2].invoke({"day": "sunday"})
except Exception as error:
    print("  get_timetable('sunday')      -> rejected before the function ran:")
    print("     ", str(error).splitlines()[0])