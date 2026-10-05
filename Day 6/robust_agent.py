"""Day 6: the Day 3 agent, ported to the OpenAI-compatible API, with validation and retry."""
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "Day 1"))
from config import client, MODEL, banner
from tools_v2 import TOOLS, TOOL_FUNCTIONS, SCHEMAS
from validate import validate_arguments

SYSTEM_PROMPT = (
    "You are a college fee assistant. Never guess a fee: always call get_course_fee. "
    "Use calculator for every arithmetic step. Valid course codes: CS101, AI202, DS303. "
    "If no tool is needed, answer directly."
)

MAX_TOKENS = 500          # raised automatically when a reply is truncated
REPEAT_LIMIT = 3          # identical calls allowed before we stop

def handle_tool_call(call, log=True):
    """Run one tool call defensively. Always returns a string for the model."""
    name = call.function.name
    raw = call.function.arguments or "{}"

    # 1. the arguments are generated text: they may not be JSON at all
    try:
        arguments = json.loads(raw)
    except json.JSONDecodeError as error:
        return f"Argument error: invalid JSON ({error}). Send valid JSON for '{name}'."

    # 2. the tool may not exist
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return f"Unknown tool: {name}. Available tools: {', '.join(TOOL_FUNCTIONS)}."

    # 3. the arguments may not match the schema
    problem = validate_arguments(arguments, SCHEMAS[name])
    if problem:
        return f"Argument error: {problem}"

    # 4. the tool itself may fail
    try:
        result = str(function(**arguments))
    except Exception as error:
        result = f"Tool error in {name}: {type(error).__name__}: {error}"

    if log:
        print(f"      {name}({arguments}) -> {result[:100]}")
    return result

def agent(question, max_steps=6, verbose=True):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    seen = {}
    max_tokens = MAX_TOKENS

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS,
            temperature=0, max_tokens=max_tokens)
        choice = response.choices[0]
        message = choice.message

        # truncated replies are a settings problem, not a model problem
        if choice.finish_reason == "length":
            if max_tokens >= 2000:
                return "Stopped: the reply was still truncated at 2000 tokens."
            max_tokens *= 2
            if verbose:
                print(f"   step {step}: truncated, retrying with max_tokens={max_tokens}")
            continue

        if not message.tool_calls:
            return (message.content or "").strip()

        messages.append({
            "role": "assistant", "content": message.content or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.function.name,
                                         "arguments": c.function.arguments}}
                           for c in message.tool_calls]})

        if verbose:
            print(f"   step {step}: {len(message.tool_calls)} tool call(s)")

        for call in message.tool_calls:                 # ALWAYS loop: there may be several
            signature = (call.function.name, call.function.arguments)
            seen[signature] = seen.get(signature, 0) + 1
            if seen[signature] >= REPEAT_LIMIT:
                return (f"Stopped: {call.function.name} was called {REPEAT_LIMIT} times "
                        f"with the same arguments and made no progress.")

            result = handle_tool_call(call, log=verbose)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    return "Stopped: maximum steps reached without a final answer."

if __name__ == "__main__":
    banner("ROBUST AGENT")
    for question in [
        "What is the total fee for CS101 and AI202 after a 10% scholarship?",
        "Is DS303 more expensive than CS101, and by how much?",
        "What is the fee for ME404?",
        "Write a one-line welcome message for new students.",
    ]:
        print("\nQ:", question)
        print("A:", agent(question))