"""Day 7, Part A: prompt | model | output parser."""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from lc_config import get_model, PROVIDER

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Answer in at most {limit} words."),
    ("human", "{question}"),
])
model = get_model()
parser = StrOutputParser()

chain = prompt | model | parser          # the whole of LCEL in one line

if __name__ == "__main__":
    print(f"=== LCEL CHAIN | provider: {PROVIDER} ===\n")

    # 1. invoke: one input, one output
    print("[invoke]")
    print(chain.invoke({"role": "college counsellor", "limit": 25,
                        "question": "Why should a student learn about AI agents?"}))

    # 2. batch: several inputs, the SAME chain, no extra code
    print("\n[batch]")
    answers = chain.batch([
        {"role": "physics teacher", "limit": 15, "question": "What is gravity?"},
        {"role": "chef",            "limit": 15, "question": "What is gravity?"},
        {"role": "poet",            "limit": 15, "question": "What is gravity?"},
    ])
    for role, answer in zip(["physics teacher", "chef", "poet"], answers):
        print(f"  {role:<16} {answer}")

    # 3. stream: tokens as they are produced
    print("\n[stream]")
    for piece in chain.stream({"role": "storyteller", "limit": 40,
                               "question": "Tell a two-line story about a robot in a library."}):
        print(piece, end="", flush=True)
    print()

    # 4. look inside: what does each stage actually produce?
    print("\n[inspect]")
    filled = prompt.invoke({"role": "assistant", "limit": 10, "question": "Hi"})
    print("  prompt output :", type(filled).__name__, "->", filled.to_messages())
    reply = (prompt | model).invoke({"role": "assistant", "limit": 10, "question": "Hi"})
    print("  model output  :", type(reply).__name__)
    text = parser.invoke(reply)
    print("  parser output :", "a plain string" if isinstance(text, str) else type(text).__name__)