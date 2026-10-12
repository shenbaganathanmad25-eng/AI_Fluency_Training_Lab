"""Day 7, Part C: bind the three tools to a local chat model."""
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from lc_config import get_model, PROVIDER
from lc_tools import TOOLS

tools_by_name = {t.name: t for t in TOOLS}
model_with_tools = get_model().bind_tools(TOOLS)

SYSTEM = SystemMessage(
    "You are a college assistant. Use get_course_fee for fees, calculator for any "
    "arithmetic, and get_timetable for class schedules. Never guess.")

def ask(question, max_steps=5):
    print("\nQ:", question)
    messages = [SYSTEM, HumanMessage(question)]

    for step in range(1, max_steps + 1):
        ai = model_with_tools.invoke(messages)
        messages.append(ai)

        for bad in ai.invalid_tool_calls:                 # arguments that would not parse
            print(f"   step {step}: INVALID call {bad.get('name')}: {bad.get('error')}")

        if not ai.tool_calls:
            print("A:", ai.content.strip())
            return

        for call in ai.tool_calls:
            tool = tools_by_name.get(call["name"])
            if tool is None:
                result = f"Unknown tool: {call['name']}. Available: {', '.join(tools_by_name)}"
                messages.append(ToolMessage(result, tool_call_id=call["id"]))
            else:
                result_message = tool.invoke(call)        # returns a ready-made ToolMessage
                messages.append(result_message)
                result = result_message.content
            print(f"   step {step}: {call['name']}({call['args']}) -> {result}")

    print("A: Stopped: maximum steps reached.")

if __name__ == "__main__":
    print(f"=== BIND TOOLS | provider: {PROVIDER} ===")

    # First, look at a single raw reply
    raw = model_with_tools.invoke([SYSTEM, HumanMessage("What is the fee for AI202?")])
    print("\nRaw AIMessage.tool_calls:", raw.tool_calls)

    ask("What is the total fee for CS101 and AI202 after a 10% scholarship?")
    ask("What classes do I have on Wednesday?")
    ask("Is DS303 more expensive than CS101, and what is on Thursday?")
    ask("Write a one-line welcome message for new students.")