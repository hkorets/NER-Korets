from functools import lru_cache
from src.backend.agent.agent import MyLangChainAgent
from src.backend.tools.ner import NER
from src.backend.tools.rag import RAG

@lru_cache
def get_agent() -> MyLangChainAgent:
    return MyLangChainAgent()

@lru_cache
def get_ner_service() -> NER:
    return NER()

@lru_cache
def get_rag_service() -> RAG:
    return RAG()
