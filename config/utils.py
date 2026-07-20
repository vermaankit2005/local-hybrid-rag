from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from config.constants import OPENROUTER_BASE_URL, TEXT_EMBEDDING_LARGE, LLM_MODEL
from config.setup_opensearch import OPENROUTER_API_KEY
from langchain_groq import ChatGroq


def get_embedding_model():
    """
    Returns an instance of OpenAIEmbeddings configured with the OpenRouter API key and model.
    :return: An instance of OpenAIEmbeddings.
    """
    return OpenAIEmbeddings(
        openai_api_base=OPENROUTER_BASE_URL,
        model=TEXT_EMBEDDING_LARGE,
        openai_api_key=OPENROUTER_API_KEY
    )


def get_llm_model() -> ChatOpenAI:
    return ChatOpenAI(
        openai_api_base=OPENROUTER_BASE_URL,
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        temperature=0.4,
        reasoning_effort="medium",
        max_retries=3,
    )

def get_grok_llm():
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.4, reasoning_format="hidden")
