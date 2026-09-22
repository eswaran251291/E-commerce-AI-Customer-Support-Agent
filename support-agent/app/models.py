from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Intent(str, Enum):
    ORDER_STATUS = "order_status"
    TRACK_DELIVERY = "track_delivery"
    RETURN_REQUEST = "return_request"
    REFUND_STATUS = "refund_status"
    CANCEL_ORDER = "cancel_order"
    PRODUCT_DAMAGE = "product_damage"
    POLICY_QUESTION = "policy_question"
    ACCOUNT_PAYMENT = "account_payment"
    GREETING = "greeting"
    UNKNOWN = "unknown"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Entities(BaseModel):
    order_id: str | None = None
    tracking_number: str | None = None
    email: str | None = None
    zip_code: str | None = None
    phone_last4: str | None = None
    product: str | None = None


class Understanding(BaseModel):
    intent: Intent = Intent.UNKNOWN
    confidence: float = 0.0
    entities: Entities = Field(default_factory=Entities)
    sentiment: Sentiment = Sentiment.NEUTRAL
    source: Literal["rules", "llm"] = "rules"


class Action(BaseModel):
    """A side effect the agent performed on a backend system."""

    type: str
    detail: str
    reference: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class Escalation(BaseModel):
    ticket_id: str
    reason: str
    trigger: str
    urgency: Urgency
    sentiment: Sentiment
    customer_id: str | None = None
    order_id: str | None = None
    attempted_solutions: list[str] = Field(default_factory=list)
    transcript: list[dict[str, str]] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    understanding: Understanding
    actions: list[Action] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    escalated: bool = False
    escalation: Escalation | None = None
    identity_verified: bool = False
