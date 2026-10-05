"""Day 6: check tool arguments against the JSON Schema before calling the function."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "Day 1"))
import config

TYPES = {"string": str, "number": (int, float), "integer": int,
         "boolean": bool, "array": list, "object": dict}

def validate_arguments(arguments, schema):
    """Return None when the arguments are valid, or an error message explaining why not."""
    if not isinstance(arguments, dict):
        return "Arguments must be a JSON object."

    properties = schema.get("properties", {})

    for name in schema.get("required", []):                 # 1. missing required
        if name not in arguments:
            return (f"Missing required argument '{name}'. "
                    f"Expected: {', '.join(properties)}.")

    if schema.get("additionalProperties") is False:         # 2. invented extras
        extra = [k for k in arguments if k not in properties]
        if extra:
            return (f"Unexpected argument(s): {', '.join(extra)}. "
                    f"Allowed: {', '.join(properties)}.")

    for name, value in arguments.items():                   # 3. types and enums
        rule = properties.get(name, {})
        expected = TYPES.get(rule.get("type"))
        if expected and not isinstance(value, expected):
            return (f"Argument '{name}' must be a {rule['type']}, "
                    f"but got {type(value).__name__}: {value!r}.")
        if "enum" in rule and value not in rule["enum"]:
            return (f"Argument '{name}' must be one of {rule['enum']}, got {value!r}.")

    return None

if __name__ == "__main__":
    from tools_v2 import SCHEMAS
    schema = SCHEMAS["get_course_fee"]
    cases = [
        {"course_code": "CS101"},
        {"course_code": "CS101", "semester": "even"},
        {},
        {"course_code": 101},
        {"course_code": "CS101", "semester": "summer"},
        {"course_code": "CS101", "year": 2026},
    ]
    for case in cases:
        print(f"{str(case):<48} -> {validate_arguments(case, schema) or 'OK'}")