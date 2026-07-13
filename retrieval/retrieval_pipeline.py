from IPython.core.display_functions import display
from IPython.display import Markdown
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from config.setup_opensearch import get_llm_model
from retrieval.document_retrieval import DocumentRetrieval

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer using only the provided context. "
     "If the context doesn't contain the answer, say you don't know. "
     "Reply strictly in markdown format."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def _retrieve(question: str) -> str:
    docs = DocumentRetrieval.hybrid_search(question, num_documents=20)
    return "\n\n---\n\n".join(docs)


rag_chain = (
        {
            "context": RunnableLambda(lambda x: _retrieve(x["question"])),
            "question": RunnablePassthrough() | RunnableLambda(lambda x: x["question"]),
        }
        | PROMPT
        | get_llm_model()
        | StrOutputParser()
)


def query(question: str) -> str:
    return rag_chain.invoke({"question": question})


if __name__ == "__main__":
    display(Markdown(query("How does mickey mouse look like? Explain in 1 paragraph. Include all details")).data)
