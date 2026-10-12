
"""Day 8: Agentic RAG Helpdesk in LangGraph."""

import ast
import operator

from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from lc_config import get_model, get_vectorstore


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

COURSE_FEES = {
    "CS101": 12000,
    "AI202": 18000,
    "DS303": 15000,
}

store = get_vectorstore()

# IMPORTANT:
# Measure your actual placement and France scores first.
# Change this threshold based on your measured scores.
MAX_DISTANCE = 0.6


# --------------------------------------------------
# 2. SEARCH HANDBOOK TOOL + RELEVANCE GUARD
# --------------------------------------------------

@tool
def search_handbook(query: str) -> str:
    """Search college handbook documents for relevant rules and policies."""

    results = store.similarity_search_with_score(query, k=3)

    # Keep only relevant chunks.
    relevant_docs = [
        (doc, score)
        for doc, score in results
        if score <= MAX_DISTANCE
    ]

    # No relevant handbook content found.
    if not relevant_docs:
        return (
            "NO_MATCH: this is not covered in the college handbook."
        )

    # Return relevant text and source filenames.
    return "\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}] "
        f"{doc.page_content}"
        for doc, score in relevant_docs
    )


# --------------------------------------------------
# 3. COURSE FEE TOOL
# --------------------------------------------------

@tool
def get_course_fee(course_code: str) -> str:
    """Return the fee in rupees for a course code such as CS101, AI202 or DS303."""

    code = course_code.strip().upper()
    fee = COURSE_FEES.get(code)

    if fee is not None:
        return f"{code} fee is Rs. {fee}"

    return f"Unknown course code {course_code}"


# --------------------------------------------------
# 4. CALCULATOR TOOL
# --------------------------------------------------

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


@tool
def calculator(expression: str) -> str:
    """Evaluate simple arithmetic such as '18000 + 15000' or '100 * 12'."""

    def ev(node):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](
                ev(node.left),
                ev(node.right),
            )

        raise ValueError("Only +, -, * and / on numbers are allowed")

    try:
        expression_tree = ast.parse(expression, mode="eval")
        return str(ev(expression_tree.body))

    except Exception as error:
        return f"Error: {error}"


# --------------------------------------------------
# 5. NEW DAY 8 TOOL: EXAM ELIGIBILITY
# --------------------------------------------------

@tool
def check_exam_eligibility(attendance_percent: float) -> str:
    """Check if a student with this attendance percentage (0-100) may write the end-semester exam."""

    # Reject values outside the valid range.
    if attendance_percent < 0 or attendance_percent > 100:
        return "ERROR: Attendance must be between 0 and 100."

    # 75% or above.
    if attendance_percent >= 75:
        return "ELIGIBLE: You may write the end-semester exam."

    # 65% to below 75%.
    elif attendance_percent >= 65:
        return (
            "CONDONATION: You need condonation. "
            "Pay Rs. 500 per course."
        )

    # Below 65%.
    else:
        return (
            "NOT ELIGIBLE: Attendance below 65% "
            "does not meet exam eligibility requirements."
        )


# --------------------------------------------------
# 6. REGISTER ALL TOOLS
# --------------------------------------------------

tools = [
    search_handbook,
    get_course_fee,
    calculator,
    check_exam_eligibility,
]


# --------------------------------------------------
# 7. MODEL AND SYSTEM PROMPT
# --------------------------------------------------

model = get_model().bind_tools(tools)

SYSTEM = """
You are the Greenfield College helpdesk assistant.

Follow these rules:

1. Use search_handbook for college rules, policies,
   attendance regulations and late-fee information.

2. If search_handbook returns NO_MATCH, say:
   "I don't know. This is not covered in the college handbook."
   Do not guess or invent an answer.

3. Use get_course_fee to retrieve course fees.

4. Use calculator for arithmetic.
   Do not calculate multi-step totals mentally.

5. Use check_exam_eligibility whenever a student asks
   whether they can write the end-semester exam based
   on their attendance percentage.

6. Follow the eligibility tool's result exactly:
   - 75% or above: eligible.
   - 65% to below 75%: condonation, Rs. 500 per course.
   - Below 65%: not eligible.
   - Outside 0-100: error.

7. For follow-up questions, use the conversation history
   when relevant. For example, "And if I pay only 5 days
   late instead?" refers to the previous fee question.

8. Answer briefly and identify the source file when
   handbook information is used.

9. Never invent college policies, fees or late-fee rules.
"""


# --------------------------------------------------
# 8. LANGGRAPH AGENT NODE
# --------------------------------------------------

def agent(state: MessagesState):
    """Read the conversation and either answer or call a tool."""

    reply = model.invoke(
        [("system", SYSTEM)] + state["messages"]
    )

    return {"messages": [reply]}


# --------------------------------------------------
# 9. BUILD GRAPH WITH CONVERSATION MEMORY
# --------------------------------------------------

builder = StateGraph(MessagesState)

builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")

# Tool calls go to the tools node; normal answers end.
builder.add_conditional_edges("agent", tools_condition)

# After tool execution, return to the agent.
builder.add_edge("tools", "agent")

# In-memory checkpoint supports conversation threads.
graph = builder.compile(
    checkpointer=InMemorySaver()
)


# --------------------------------------------------
# 10. ASK FUNCTION: PRINT TOOL CALLS AND ANSWERS
# --------------------------------------------------

def ask(question, thread_id):
    """Ask a question using a specific conversation thread."""

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 10,
    }

    print(f"\n[{thread_id}] USER: {question}")

    for step in graph.stream(
        {"messages": [("user", question)]},
        config,
        stream_mode="updates",
    ):
        for node, update in step.items():
            for msg in update.get("messages", []):

                # Print the tool calls made by the model.
                if getattr(msg, "tool_calls", None):
                    for call in msg.tool_calls:
                        print(
                            f"   {node:6} -> call "
                            f"{call['name']}({call['args']})"
                        )

                # Print tool results.
                elif node == "tools":
                    print(
                        f"   {node:6} -> {msg.name} returned "
                        f"{str(msg.content)[:150]!r}"
                    )

                # Print assistant response.
                else:
                    print(
                        f"   {node:6} -> ANSWER: {msg.content}"
                    )


# --------------------------------------------------
# 11. PRINT SCORES FOR PART C
# --------------------------------------------------

def print_relevance_scores(query):
    """Print distances and source files for threshold selection."""

    print(f"\nQuery: {query}")

    results = store.similarity_search_with_score(query, k=3)

    if not results:
        print("No search results.")
        return

    for doc, score in results:
        print(
            f"Score: {score:.4f} | "
            f"Source: {doc.metadata.get('source', 'unknown')}"
        )


# --------------------------------------------------
# 12. TEST ALL FIVE DAY 8 QUESTIONS
# --------------------------------------------------

if __name__ == "__main__":

    # Print measured scores to help select MAX_DISTANCE.
    print("\n========== PART C: RELEVANCE SCORES ==========")

    print_relevance_scores(
        "What CGPA do I need to be eligible for placements?"
    )

    print_relevance_scores(
        "What is the capital of France?"
    )

    print(f"\nCurrent MAX_DISTANCE = {MAX_DISTANCE}")
    print("Adjust it based on your measured scores.")

    # Show the workflow diagram in Mermaid format.
    print("\n========== LANGGRAPH WORKFLOW ==========")
    print(graph.get_graph().draw_mermaid())

    # All five questions use the SAME thread.
    thread_id = "task-run"

    print("\n========== PART D: FIVE TEST QUESTIONS ==========")

    ask(
        "What CGPA do I need to be eligible for placements?",
        thread_id,
    )

    ask(
        "My attendance is 70%. Can I write the exam?",
        thread_id,
    )

    ask(
        "What is the total of the CS101 fee, the AI202 fee "
        "and the maximum late fee?",
        thread_id,
    )

    ask(
        "And if I pay only 5 days late instead?",
        thread_id,
    )

    ask(
        "What is the capital of France?",
        thread_id,
    )

    # Show how many messages were saved in the thread.
    saved = graph.get_state(
        {"configurable": {"thread_id": thread_id}}
    ).values["messages"]

    print(
        f"\nThread {thread_id} has "
        f"{len(saved)} messages saved in memory."
    )
