from __future__ import annotations

import re

from .models import Entities, Intent, Sentiment, Understanding

ORDER_ID_RE = re.compile(r"\bORD[- ]?(\d{4,6})\b", re.IGNORECASE)
TRACKING_RE = re.compile(r"\b(?:1Z[0-9A-Z]{16}|BD\d{9}[A-Z]{2}|\d{4}\s?\d{4}\s?\d{4})\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
ZIP_RE = re.compile(
    r"\b(?:zip|postal|pin)\s*(?:code)?\s*(?:is|=)?\s*[:#]?\s*(\d{5,6})\b", re.IGNORECASE
)
PHONE4_RE = re.compile(r"\b(?:phone|mobile|number)\D{0,20}(\d{4})\b", re.IGNORECASE)

INTENT_KEYWORDS: dict[Intent, list[str]] = {
    Intent.PRODUCT_DAMAGE: [
        "damaged", "broken", "cracked", "defective", "faulty", "doesn't work",
        "does not work", "stopped working", "torn", "scratched", "poor quality",
        "missing part", "leaking",
    ],
    Intent.RETURN_REQUEST: ["return", "send it back", "send this back", "exchange", "rma", "return label"],
    Intent.REFUND_STATUS: ["refund", "money back", "reimburse", "when will i be credited", "charge reversed"],
    Intent.CANCEL_ORDER: ["cancel", "call off my order", "don't ship", "do not ship"],
    Intent.TRACK_DELIVERY: [
        "track", "tracking", "where is my", "where's my", "delivery", "delivered",
        "shipment", "courier", "arrive", "eta", "late", "delayed",
    ],
    Intent.ORDER_STATUS: ["order status", "status of my order", "my order", "order update", "has it shipped"],
    Intent.ACCOUNT_PAYMENT: [
        "payment", "charged", "charged twice", "card", "billing", "invoice",
        "change my address", "update address", "my account", "password",
    ],
    Intent.POLICY_QUESTION: [
        "policy", "how long do i have", "return window", "warranty", "how long does",
        "do you ship", "shipping cost", "what is your", "can i return",
    ],
    Intent.GREETING: ["hi", "hello", "hey", "good morning", "good evening", "thanks", "thank you"],
}

# Checked in order; first match with a keyword hit wins.
INTENT_PRIORITY = [
    Intent.PRODUCT_DAMAGE,
    Intent.CANCEL_ORDER,
    Intent.RETURN_REQUEST,
    Intent.REFUND_STATUS,
    Intent.TRACK_DELIVERY,
    Intent.ORDER_STATUS,
    Intent.ACCOUNT_PAYMENT,
    Intent.POLICY_QUESTION,
    Intent.GREETING,
]

FRUSTRATION_MARKERS = [
    "frustrated", "annoyed", "unacceptable", "ridiculous", "third time", "again and again",
    "still waiting", "no one", "nobody", "useless", "terrible", "worst", "fed up", "poor service",
]
ANGER_MARKERS = ["angry", "furious", "outrageous", "lawyer", "sue", "scam", "never again", "cancel my account"]
POSITIVE_MARKERS = ["thanks", "thank you", "great", "appreciate", "awesome", "perfect"]


def extract_entities(text: str) -> Entities:
    entities = Entities()
    if match := ORDER_ID_RE.search(text):
        entities.order_id = f"ORD-{match.group(1)}"
    if match := TRACKING_RE.search(text):
        entities.tracking_number = match.group(0).strip()
    if match := EMAIL_RE.search(text):
        entities.email = match.group(0)
    if match := ZIP_RE.search(text):
        entities.zip_code = match.group(1)
    if match := PHONE4_RE.search(text):
        entities.phone_last4 = match.group(1)
    return entities


def detect_sentiment(text: str) -> Sentiment:
    lowered = text.lower()
    if any(marker in lowered for marker in ANGER_MARKERS) or text.count("!") >= 2:
        return Sentiment.ANGRY
    if any(marker in lowered for marker in FRUSTRATION_MARKERS):
        return Sentiment.FRUSTRATED
    if any(marker in lowered for marker in POSITIVE_MARKERS):
        return Sentiment.POSITIVE
    return Sentiment.NEUTRAL


def classify_intent(text: str) -> tuple[Intent, float]:
    lowered = text.lower()
    if not ORDER_ID_RE.search(text):
        policy_hits = [kw for kw in INTENT_KEYWORDS[Intent.POLICY_QUESTION] if kw in lowered]
        if policy_hits:
            return Intent.POLICY_QUESTION, min(0.55 + 0.15 * len(policy_hits), 0.95)
    for intent in INTENT_PRIORITY:
        hits = [kw for kw in INTENT_KEYWORDS[intent] if kw in lowered]
        if hits:
            confidence = min(0.55 + 0.15 * len(hits), 0.95)
            if intent is Intent.GREETING and len(lowered.split()) > 6:
                continue
            return intent, confidence
    if ORDER_ID_RE.search(text):
        return Intent.ORDER_STATUS, 0.5
    return Intent.UNKNOWN, 0.2


def understand(text: str) -> Understanding:
    intent, confidence = classify_intent(text)
    return Understanding(
        intent=intent,
        confidence=confidence,
        entities=extract_entities(text),
        sentiment=detect_sentiment(text),
        source="rules",
    )
