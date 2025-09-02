from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from src.backend.api.deps import get_agent
from src.backend.agent.agent import MyLangChainAgent
import traceback

router = APIRouter(prefix="/agent", tags=["agent"])

class RunRequest(BaseModel):
    input: str

class RunResponse(BaseModel):
    output: str

@router.post("/run", response_model=RunResponse)
def run_agent(payload: RunRequest, agent: MyLangChainAgent = Depends(get_agent)):
    try:
        output = agent.run(payload.input)
        return RunResponse(output=output)
    except Exception as e:
        # лог у консоль для дебагу
        print("AGENT ERROR:\n", traceback.format_exc())
        # віддати текст помилки клієнту
        raise HTTPException(status_code=500, detail=str(e))
