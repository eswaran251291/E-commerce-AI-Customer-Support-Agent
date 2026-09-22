from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .agent import SupportAgent
from .models import ChatRequest, ChatResponse, Escalation

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="E-commerce Support Agent", version="1.0.0")
agent = SupportAgent()


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "llm_enabled": agent.llm.enabled,
        "model": agent.llm.model if agent.llm.enabled else None,
        "orders_loaded": len(agent.store.orders),
        "kb_articles": len(agent.store.kb["articles"]),
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    return agent.handle(request.session_id, request.message)


@app.post("/api/sessions/{session_id}/reset")
def reset(session_id: str) -> dict[str, str]:
    agent.sessions.pop(session_id, None)
    return {"session_id": session_id, "status": "reset"}


@app.get("/api/orders/{order_id}")
def get_order(order_id: str) -> dict[str, object]:
    order = agent.store.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="order not found")
    return order


@app.get("/api/orders")
def list_orders() -> dict[str, object]:
    return {
        "orders": [
            {
                "order_id": o["order_id"],
                "status": o["status"],
                "customer": agent.store.customers[o["customer_id"]]["name"],
                "expected_delivery": o["expected_delivery"],
            }
            for o in agent.store.orders.values()
        ]
    }


@app.get("/api/escalations", response_model=list[Escalation])
def escalations() -> list[Escalation]:
    return list(agent.store.escalations.values())


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
