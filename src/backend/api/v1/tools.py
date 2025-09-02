from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.backend.api.deps import get_ner_service, get_rag_service
from src.backend.tools.ner import NER
from src.backend.tools.rag import RAG

router = APIRouter(prefix="/tools", tags=["tools"])

class AnonymizeRequest(BaseModel):
    text: str

class RagAskRequest(BaseModel):
    query: str

@router.post("/anonymize_text")
def anonymize_text(payload: AnonymizeRequest, ner: NER = Depends(get_ner_service)):
    return {"result": ner.anonymize(payload.text)}

@router.post("/rag_ask")
def rag_ask(payload: RagAskRequest, rag: RAG = Depends(get_rag_service)):
    return {"result": rag.answer_question(payload.query)}
