"""Step 5: an agentic RAG helpdesk in LangGraph - tools + loop + memory."""
import ast
import operator

from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from lc_config import get_model, get_vectorstore

COURSE_FEES = {"CS101": 12000, "AI202": 18000, "DS303": 15000}   # same data as Day 1
store = get_vectorstore()


@tool
def search_handbook(query: str) -> str:
    """Search the college handbook (fees, hostel, exams, library, AI training) and return matching passages."""
    docs = store.similarity_search(query, k=3)
    return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in docs)


@tool
def get_course_fee(course_code: str) -> str:
    """Return the fee in rupees for a course code such as CS101, AI202 or DS303."""
    fee = COURSE_FEES.get(course_code.strip().upper())
    return f"{course_code.upper()} fee is Rs. {fee}" if fee else f"Unknown course code {course_code}"


OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


@tool
def calculator(expression: str) -> str:
    """Evaluate simple arithmetic such as '18000 + 15000' or '100 * 12'."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError("only + - * / on numbers")
    try:
        return str(ev(ast.parse(expression, mode="eval").body))
    except Exception as e:                       # tell the model, don't crash the graph
        return f"Error: {e}"



tools = [search_handbook, get_course_fee, calculator]
model = get_model().bind_tools(tools)
SYSTEM = ("You are the Greenfield College helpdesk. Use search_handbook for any rule or policy, "
          "get_course_fee for course fees and calculator for arithmetic. "
          "Answer briefly and name the source file. If the tools do not give the answer, say you don't know.")


def agent(state: MessagesState):
    """The LLM node: read the conversation, either answer or ask for a tool."""
    reply = model.invoke([("system", SYSTEM)] + state["messages"])
    return {"messages": [reply]}


builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))          # runs every tool call in the last message
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)   # tool calls? -> "tools", else -> END
builder.add_edge("tools", "agent")                  # the LOOP back to the LLM
graph = builder.compile(checkpointer=InMemorySaver())  # checkpointer = memory per thread


def ask(question, thread_id):
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}
    print(f"\n[{thread_id}] USER: {question}")
    for step in graph.stream({"messages": [("user", question)]}, config, stream_mode="updates"):
        for node, update in step.items():
            for msg in update["messages"]:
                if getattr(msg, "tool_calls", None):
                    for c in msg.tool_calls:
                        print(f"   {node:6} -> call {c['name']}({c['args']})")
                elif node == "tools":
                    print(f"   {node:6} -> {msg.name} returned {msg.content[:55]!r}...")
                else:
                    print(f"   {node:6} -> ANSWER: {msg.content}")


if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())
    ask("What is the late fee per day if I pay my fees late?", "student-1")
    ask("And what is the maximum?", "student-1")                  # needs memory of turn 1
    ask("What is the total fee for AI202 and DS303?", "student-1")
    ask("And what is the maximum?", "student-2")                  # new thread: no memory

    for t in ("student-1", "student-2"):                          # what the checkpointer saved
        saved = graph.get_state({"configurable": {"thread_id": t}}).values["messages"]
        print(f"\nThread {t} has {len(saved)} messages saved in memory")