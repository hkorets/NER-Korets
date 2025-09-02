import json, sys
from typing import Any, Dict
from src.backend.tools.ner import NER
from src.backend.tools.rag import RAG

ner = NER()
rag = RAG()

def _respond(_id, result=None, error=None):
    msg = {"jsonrpc":"2.0","id":_id}
    if error: msg["error"] = error
    else:     msg["result"] = result
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()

def _list_tools():
    return {
        "tools": [
            {
                "name": "rag_ask",
                "description": "Answer a question using the RAG pipeline.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "anonymize_text",
                "description": "Mask entities (PII) in the input text.",
                "input_schema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        ]
    }

def _call_tool(name: str, arguments: Dict[str, Any]):
    if name == "rag_ask":
        query = arguments["query"]
        return {"content": {"answer": rag.answer_question(query)}}
    if name == "anonymize_text":
        text = arguments["text"]
        return {"content": {"text": ner.anonymize(text)}}
    raise ValueError(f"Unknown tool: {name}")

def _handle(req: Dict[str, Any]):
    _id = req.get("id")
    method = req.get("method")
    params = req.get("params", {})

    try:
        if method == "tools/list":
            _respond(_id, _list_tools())
        elif method == "tools/call":
            name = params["name"]
            args = params.get("arguments", {})
            _respond(_id, _call_tool(name, args))
        else:
            _respond(_id, error={"code": -32601, "message": "Method not found"})
    except Exception as e:
        _respond(_id, error={"code": -32000, "message": str(e)})

def main():
    for line in sys.stdin:
        if line.strip():
            _handle(json.loads(line))

if __name__ == "__main__":
    main()
