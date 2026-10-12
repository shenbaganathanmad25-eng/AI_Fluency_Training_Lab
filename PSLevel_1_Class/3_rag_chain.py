"""Step 3: Retrieval-Augmented Generation as one LCEL chain."""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from lc_config import get_model, get_vectorstore

retriever = get_vectorstore().as_retriever(search_kwargs={"k": 3})


def format_docs(docs):
    """Join the retrieved chunks into one block, each tagged with its file name."""
    return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)


prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are the college helpdesk. Answer ONLY from the context below. "
     "If the answer is not in the context, say \"I don't know - please contact the office.\" "
     "End with 'Source: <file name>'.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

rag_chain = (
    {"context": retriever | RunnableLambda(format_docs),   # question -> chunks -> text
     "question": RunnablePassthrough()}                     # question passes through unchanged
    | prompt
    | get_model()
    | StrOutputParser()
)

if __name__ == "__main__":
    for q in ["What is the maximum late fee in a semester?",
              "Are electric kettles allowed in the hostel?",
              "Who won the cricket world cup?"]:          # not in the handbook
        print(f"Q: {q}")
        print("A:", rag_chain.invoke(q), "\n")