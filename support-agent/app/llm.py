from __future__ import annotations

import json
import os
from typing import Any

from .models import Entities, Intent, Sentiment, Understanding

SYSTEM_NLU = """You classify e-commerce customer support messages.
Return strict JSON: {"intent": one of [order_status, track_delivery, return_request, refund_status,
cancel_order, product_damage, policy_question, account_payment, greeting, unknown],
"confidence": 0-1, "sentiment": one of [positive, neutral, frustrated, angry],
"entities": {"order_id": str|null, "tracking_number": str|null, "email": str|null,
"zip_code": str|null, "phone_last4": str|null, "product": str|null}}"""

SYSTEM_PHRASE = """You are an e-commerce customer support agent. Rewrite the draft reply so it is warm,
empathetic and natural. Rules: use ONLY the supplied facts and draft - never invent order numbers, dates,
amounts or policies; keep it to 2-3 sentences; start with the direct answer; end with a clarifying question
or an offer of further help; no bullet points, no robotic phrasing."""


class LLMClient:
    """Optional LLM layer. The agent is fully functional without it.

    When OPENAI_API_KEY is set, the LLM is used for intent/entity extraction and for
    phrasing the final reply on top of facts the deterministic core retrieved.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("SUPPORT_AGENT_MODEL", "gpt-4o-mini")
        self._client: Any | None = None
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=api_key)
            except Exception:
                self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def _chat(self, system: str, user: str, json_mode: bool = False) -> str | None:
        if self._client is None:
            return None
        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.3,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            completion = self._client.chat.completions.create(**kwargs)
            return completion.choices[0].message.content
        except Exception:
            return None

    def understand(self, message: str) -> Understanding | None:
        raw = self._chat(SYSTEM_NLU, message, json_mode=True)
        if not raw:
            return None
        try:
            data = json.loads(raw)
            return Understanding(
                intent=Intent(data.get("intent", "unknown")),
                confidence=float(data.get("confidence", 0.6)),
                sentiment=Sentiment(data.get("sentiment", "neutral")),
                entities=Entities(**{k: v for k, v in (data.get("entities") or {}).items() if v}),
                source="llm",
            )
        except Exception:
            return None

    def phrase(
        self,
        message: str,
        draft: str,
        facts: list[str],
        transcript: list[dict[str, str]] | None = None,
    ) -> str:
        if self._client is None:
            return draft
        history = "\n".join(f"{t['role']}: {t['content']}" for t in (transcript or [])[-6:])
        user = (
            f"Conversation so far:\n{history}\n\n"
            f"Customer message: {message}\n\n"
            f"Verified facts:\n- " + "\n- ".join(facts or ["none"]) + f"\n\nDraft reply:\n{draft}"
        )
        return self._chat(SYSTEM_PHRASE, user) or draft
