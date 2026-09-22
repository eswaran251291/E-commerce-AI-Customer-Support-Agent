from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agent import SupportAgent
from app.llm import LLMClient
from app.main import app
from app.models import Intent, Sentiment
from app.nlu import understand


class NoLLM(LLMClient):
    def __init__(self) -> None:  # bypass env detection so tests are deterministic
        self.model = "rules"
        self._client = None


@pytest.fixture()
def agent() -> SupportAgent:
    return SupportAgent(llm=NoLLM())


def test_intent_and_entity_extraction() -> None:
    result = understand("Where is my order ORD-10234? It's already late")
    assert result.intent is Intent.TRACK_DELIVERY
    assert result.entities.order_id == "ORD-10234"


def test_damage_report_escalates_with_context(agent: SupportAgent) -> None:
    response = agent.handle("s1", "The espresso machine in ORD-10236 arrived cracked")
    assert response.escalated
    assert response.escalation is not None
    assert response.escalation.trigger == "product_quality_or_damage"
    assert response.escalation.order_id == "ORD-10236"
    assert response.escalation.transcript


def test_policy_exception_escalates(agent: SupportAgent) -> None:
    response = agent.handle("s2", "Can you make an exception and take back ORD-10236?")
    assert response.escalated
    assert response.escalation.trigger == "policy_exception_request"


def test_repeated_dissatisfaction_escalates(agent: SupportAgent) -> None:
    agent.handle("s3", "This is unacceptable, I am still waiting")
    second = agent.handle("s3", "Still nobody has helped me, this is terrible")
    assert second.escalated
    assert second.escalation.trigger == "repeated_dissatisfaction"
    assert second.escalation.sentiment in (Sentiment.FRUSTRATED, Sentiment.ANGRY)


def test_order_details_require_verification(agent: SupportAgent) -> None:
    first = agent.handle("s4", "What's the status of ORD-10234?")
    assert not first.identity_verified
    assert "confirm" in first.reply.lower()

    second = agent.handle("s4", "my zip code is 600042")
    assert second.identity_verified
    assert "BD884512339IN" in second.reply


def test_verification_does_not_carry_to_another_customers_order(agent: SupportAgent) -> None:
    agent.handle("s11", "status of ORD-10234? zip 600042")
    response = agent.handle("s11", "and where is ORD-10236?")
    assert not response.identity_verified
    assert "confirm" in response.reply.lower()
    assert "FedEx" not in response.reply


def test_wrong_verification_detail_keeps_order_locked(agent: SupportAgent) -> None:
    agent.handle("s12", "status of ORD-10234?")
    response = agent.handle("s12", "zip 99999")
    assert not response.identity_verified
    assert "BD884512339IN" not in response.reply


def test_delayed_order_offers_compensation(agent: SupportAgent) -> None:
    agent.handle("s5", "Tracking for ORD-10234 please, zip 600042")
    response = agent.handle("s5", "any update on ORD-10234?")
    voucher = [a for a in response.actions if a.type == "issue_voucher"]
    assert voucher and voucher[0].reference


def test_return_within_window_creates_rma(agent: SupportAgent) -> None:
    agent.handle("s6", "I want to return ORD-10235")
    response = agent.handle("s6", "zip 10021")
    rma = [a for a in response.actions if a.type == "create_return"]
    assert rma, response.reply
    assert rma[0].reference.startswith("RMA-")


def test_return_outside_window_is_refused_politely(agent: SupportAgent) -> None:
    agent.handle("s7", "I need to return ORD-10236")
    response = agent.handle("s7", "zip 94107")
    assert not response.escalated
    assert "window" in response.reply.lower()
    assert not [a for a in response.actions if a.type == "create_return"]


def test_refund_status_for_pending_order(agent: SupportAgent) -> None:
    agent.handle("s8", "refund update on ORD-10238?")
    response = agent.handle("s8", "zip 10021")
    assert "refund" in response.reply.lower()


def test_policy_question_answered_from_kb(agent: SupportAgent) -> None:
    response = agent.handle("s9", "How long do I have to return electronics?")
    assert response.citations
    assert not response.escalated


def test_no_order_id_prompts_for_one(agent: SupportAgent) -> None:
    response = agent.handle("s10", "where is my package")
    assert "order id" in response.reply.lower()


def test_api_chat_and_escalations_endpoints() -> None:
    client = TestClient(app)
    assert client.get("/api/health").json()["status"] == "ok"
    payload = {"session_id": "api-1", "message": "My ORD-10236 item is broken"}
    body = client.post("/api/chat", json=payload).json()
    assert body["escalated"] is True
    assert any(e["ticket_id"] == body["escalation"]["ticket_id"] for e in client.get("/api/escalations").json())
    assert client.post("/api/chat", json={"session_id": "api-1", "message": "  "}).status_code == 400
    assert client.get("/api/orders/ORD-10234").json()["carrier"] == "BlueDart"
    assert client.get("/api/orders/ORD-00000").status_code == 404
