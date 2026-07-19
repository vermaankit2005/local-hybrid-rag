import queue
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from config.utils import get_llm_model
from retrieval import reranker
from retrieval.document_retrieval import OpenSearchDocumentRetrieval
from retrieval.reranker import rerank_documents

RAG_PROMPT = ChatPromptTemplate.from_messages([

    ("system",
     "You are a helpful assistant. Answer using only the provided context. "
     "If the context doesn't contain the answer, say you don't know. "
     "Reply strictly in formatted markdown."),

    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

MULTI_QUERY_RAG_PROMPT = ChatPromptTemplate.from_messages([

    ("system",
     """
        You are expert in generating multiple relevant questions from a single user query.
        Given a user query, generate a list of 3-5 relevant questions that can help
        retrieve more comprehensive information from a knowledge base.
        Output the questions in a list format, each question on a new line.
     """
     ),

    ("human", "Question: {question}"),
])


def _retrieve(question: str, num_documents: int = 5) -> list[str]:
    return OpenSearchDocumentRetrieval.hybrid_search(question, num_documents=num_documents)


def standard_search(question: str) -> str:
    rag_chain = (
            {
                "context": itemgetter("question") | RunnableLambda(lambda x: "\n ----- \n".join([r for r in _retrieve(x)])),
                "question": itemgetter("question")
            }
            | RAG_PROMPT
            | get_llm_model()
            | StrOutputParser()
    )
    return rag_chain.invoke({"question": question})


def flatten_deduplicate_retrieved_docs(retrieved_docs: list[list[str]]) -> list[str]:
    unique_docs = set()
    for docs in retrieved_docs:
        for doc in docs:
            unique_docs.add(doc)

    print(f"Deduplicated and formatted retrieved documents. Total unique documents: {len(unique_docs)}")
    return list(unique_docs)

def print_question(question: str) -> str:
    print(f"User Question: {'\n'.join(q for q in question.splitlines() if q.strip())}")
    return question


def multi_query_search(question: str) -> str:
    generate_question_pipeline = (
            MULTI_QUERY_RAG_PROMPT
            | get_llm_model()
            | StrOutputParser()
            | print_question
            | RunnableLambda(lambda x: [q.strip() for q in x.splitlines() if q.strip()]))


    multi_query_rag_chain = (
            {
                "context": RunnablePassthrough() | generate_question_pipeline | (lambda queries: [_retrieve(q) for q in queries])
                           | flatten_deduplicate_retrieved_docs | RunnableLambda(lambda docs: rerank_documents(docs, question)),
                "question": itemgetter("question")
            }
            | RAG_PROMPT
            | get_llm_model()
            | StrOutputParser()
    )
    return multi_query_rag_chain.invoke({"question": question})


if __name__ == "__main__":
    print(multi_query_search("Tell me about Mickey Mouse?"))
