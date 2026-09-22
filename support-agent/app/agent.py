from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import knowledge, nlu
from .llm import LLMClient
from .models import (
    Action,
    ChatResponse,
    Entities,
    Escalation,
    Intent,
    Sentiment,
    Understanding,
    Urgency,
)
from .store import Store

POLICY_EXCEPTION_MARKERS = [
    "exception", "make an exception", "outside the window", "past the window",
    "even though it's been", "waive", "override", "one time courtesy", "beyond 30 days",
]
DISPUTE_MARKERS = ["chargeback", "dispute", "fraud", "unauthorized", "legal", "consumer court"]


@dataclass
class Session:
    session_id: str
    transcript: list[dict[str, str]] = field(default_factory=list)
    entities: Entities = field(default_factory=Entities)
    customer_id: str | None = None
    verified: bool = False
    attempted_solutions: list[str] = field(default_factory=list)
    negative_turns: int = 0
    turns: int = 0
    escalated_ticket: str | None = None
    awaiting_verification_for: str | None = None
    pending_intent: Intent | None = None


@dataclass
class Resolution:
    """Grounded result of a handler, before natural-language phrasing."""

    reply: str
    facts: list[str] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    solution_note: str | None = None


class SupportAgent:
    def __init__(self, store: Store | None = None, llm: LLMClient | None = None) -> None:
        self.store = store or Store()
        self.llm = llm or LLMClient()
        self.sessions: dict[str, Session] = {}

    # ------------------------------------------------------------------
    def session(self, session_id: str) -> Session:
        return self.sessions.setdefault(session_id, Session(session_id=session_id))

    def handle(self, session_id: str, message: str) -> ChatResponse:
        session = self.session(session_id)
        session.turns += 1
        session.transcript.append({"role": "customer", "content": message})

        understanding = self._understand(message)
        self._merge_entities(session, understanding.entities)
        understanding = self._apply_pending_intent(session, understanding)
        if understanding.sentiment in (Sentiment.FRUSTRATED, Sentiment.ANGRY):
            session.negative_turns += 1

        trigger = self._escalation_trigger(session, understanding, message)
        if trigger:
            response = self._escalate(session, understanding, trigger, message)
        else:
            resolution = self._resolve(session, understanding, message)
            if resolution.solution_note:
                session.attempted_solutions.append(resolution.solution_note)
            reply = self.llm.phrase(
                message=message,
                draft=resolution.reply,
                facts=resolution.facts,
                transcript=session.transcript,
            )
            response = ChatResponse(
                session_id=session_id,
                reply=reply,
                understanding=understanding,
                actions=resolution.actions,
                citations=resolution.citations,
                identity_verified=session.verified,
            )

        session.transcript.append({"role": "agent", "content": response.reply})
        return response

    # ------------------------------------------------------------------
    def _understand(self, message: str) -> Understanding:
        llm_result = self.llm.understand(message)
        if llm_result is not None:
            return llm_result
        return nlu.understand(message)

    @staticmethod
    def _apply_pending_intent(session: Session, understanding: Understanding) -> Understanding:
        """Resume the goal the customer stated before we paused to verify identity."""
        if session.pending_intent is None:
            return understanding
        if understanding.intent in (Intent.UNKNOWN, Intent.GREETING) or (
            understanding.intent is Intent.ORDER_STATUS and understanding.confidence < 0.6
        ):
            return understanding.model_copy(update={"intent": session.pending_intent})
        return understanding

    @staticmethod
    def _merge_entities(session: Session, new: Entities) -> None:
        for name, value in new.model_dump().items():
            if value:
                setattr(session.entities, name, value)

    def _current_order(self, session: Session) -> dict[str, Any] | None:
        order = self.store.get_order(session.entities.order_id)
        if order is None and session.entities.tracking_number:
            order = self.store.find_order_by_tracking(session.entities.tracking_number)
            if order:
                session.entities.order_id = order["order_id"]
        return order

    # --- escalation ---------------------------------------------------
    def _escalation_trigger(
        self, session: Session, understanding: Understanding, message: str
    ) -> str | None:
        lowered = message.lower()
        if understanding.intent is Intent.PRODUCT_DAMAGE:
            return "product_quality_or_damage"
        if any(marker in lowered for marker in POLICY_EXCEPTION_MARKERS):
            return "policy_exception_request"
        if any(marker in lowered for marker in DISPUTE_MARKERS):
            return "payment_dispute_or_complex_refund"
        if session.negative_turns >= 2:
            return "repeated_dissatisfaction"
        if understanding.sentiment is Sentiment.ANGRY:
            return "customer_anger"
        if understanding.intent is Intent.UNKNOWN and session.turns >= 3:
            return "outside_knowledge_base"
        return None

    def _escalate(
        self, session: Session, understanding: Understanding, trigger: str, message: str
    ) -> ChatResponse:
        order = self._current_order(session)
        urgency = (
            Urgency.HIGH
            if understanding.sentiment is Sentiment.ANGRY or trigger == "payment_dispute_or_complex_refund"
            else Urgency.MEDIUM
        )
        reasons = {
            "product_quality_or_damage": "Reported damaged or defective product",
            "policy_exception_request": "Requesting an exception to standard policy",
            "payment_dispute_or_complex_refund": "Payment dispute or complex refund",
            "repeated_dissatisfaction": "Customer dissatisfied after multiple interactions",
            "customer_anger": "Strong negative sentiment detected",
            "outside_knowledge_base": "Question outside the supported knowledge base",
        }
        escalation = Escalation(
            ticket_id=self.store.next_ticket_id(),
            reason=reasons.get(trigger, trigger),
            trigger=trigger,
            urgency=urgency,
            sentiment=understanding.sentiment,
            customer_id=(order or {}).get("customer_id"),
            order_id=(order or {}).get("order_id") or session.entities.order_id,
            attempted_solutions=list(session.attempted_solutions),
            transcript=list(session.transcript),
        )
        self.store.create_escalation(escalation)
        session.escalated_ticket = escalation.ticket_id

        draft = (
            f"I'm sorry about this — I've brought in a specialist so it gets handled properly. "
            f"Your case reference is {escalation.ticket_id}"
            + (f" and it's linked to order {escalation.order_id}." if escalation.order_id else ".")
            + " They'll reach you by email within 4 business hours with the full history already attached. "
            "Is there anything else I can look into for you in the meantime?"
        )
        facts = [
            f"escalation ticket {escalation.ticket_id}",
            f"reason: {escalation.reason}",
            f"urgency: {urgency.value}",
            "human agent responds within 4 business hours",
        ]
        reply = self.llm.phrase(message=message, draft=draft, facts=facts, transcript=session.transcript)
        return ChatResponse(
            session_id=session.session_id,
            reply=reply,
            understanding=understanding,
            actions=[
                Action(
                    type="escalate_to_human",
                    detail=escalation.reason,
                    reference=escalation.ticket_id,
                    payload=escalation.model_dump(),
                )
            ],
            escalated=True,
            escalation=escalation,
            identity_verified=session.verified,
        )

    # --- identity -------------------------------------------------------
    def _verify(self, session: Session, order: dict[str, Any]) -> bool:
        if session.verified and session.customer_id == order["customer_id"]:
            return True
        customer = self.store.get_customer(order["customer_id"]) or {}
        provided = session.entities
        checks = [
            provided.email and provided.email.lower() == customer.get("email", "").lower(),
            provided.zip_code and provided.zip_code == customer.get("zip"),
            provided.phone_last4 and provided.phone_last4 == customer.get("phone_last4"),
        ]
        if any(checks):
            session.verified = True
            session.customer_id = order["customer_id"]
            session.awaiting_verification_for = None
            return True
        return False

    def _verification_prompt(self, session: Session, order: dict[str, Any]) -> Resolution:
        session.awaiting_verification_for = order["order_id"]
        return Resolution(
            reply=(
                f"I found order {order['order_id']} — before I share the details, can you confirm the email "
                "on the account, the delivery ZIP code, or the last 4 digits of the phone number? "
                "That keeps the order information protected."
            ),
            facts=[f"order {order['order_id']} exists", "identity not yet verified"],
            solution_note="Requested identity verification",
        )

    # --- intent handlers -------------------------------------------------
    def _resolve(self, session: Session, understanding: Understanding, message: str) -> Resolution:
        intent = understanding.intent
        if intent is Intent.GREETING:
            return Resolution(
                reply=(
                    "Hi! I can help with order status, delivery tracking, returns, and refunds. "
                    "What's going on with your order today?"
                )
            )
        if intent in (Intent.POLICY_QUESTION, Intent.ACCOUNT_PAYMENT, Intent.UNKNOWN):
            return self._answer_from_kb(message, intent)

        order = self._current_order(session)
        if order is None:
            return Resolution(
                reply=(
                    "Happy to help with that — could you share the order ID (it looks like ORD-10234) "
                    "or the tracking number so I can pull up the right order?"
                ),
                facts=["no order identified yet"],
                solution_note="Asked for order ID",
            )
        if not self._verify(session, order):
            session.pending_intent = intent
            return self._verification_prompt(session, order)
        session.pending_intent = None

        if intent in (Intent.ORDER_STATUS, Intent.TRACK_DELIVERY):
            return self._delivery_update(order)
        if intent is Intent.RETURN_REQUEST:
            return self._start_return(order, message)
        if intent is Intent.REFUND_STATUS:
            return self._refund_update(order)
        if intent is Intent.CANCEL_ORDER:
            return self._cancel(order)
        return self._answer_from_kb(message, intent)

    def _answer_from_kb(self, message: str, intent: Intent) -> Resolution:
        articles = knowledge.search(self.store.kb, message)
        if not articles:
            return Resolution(
                reply=(
                    "I want to make sure I get this right — could you tell me a bit more about what you need, "
                    "or share your order ID? I can help with orders, delivery, returns, refunds, and billing."
                ),
                facts=["no knowledge base match"],
                solution_note="No KB match found",
            )
        top = articles[0]
        return Resolution(
            reply=f"{top['body']} Does that cover what you needed, or shall I check a specific order for you?",
            facts=[f"{a['id']}: {a['body']}" for a in articles],
            citations=[a["id"] for a in articles],
            solution_note=f"Answered from knowledge base ({top['id']})",
        )

    def _delivery_update(self, order: dict[str, Any]) -> Resolution:
        status = order["status"].replace("_", " ")
        actions: list[Action] = []
        facts = [
            f"order {order['order_id']} status {status}",
            f"carrier {order.get('carrier') or 'not assigned yet'}",
            f"expected delivery {order['expected_delivery']}",
        ]
        if order["status"] == "delivered":
            reply = (
                f"Order {order['order_id']} was delivered on {order['delivered_at']} to "
                f"{order['shipping_address']} via {order['carrier']}. "
                "If it isn't with you, I can open a carrier trace right away — want me to?"
            )
            facts.append(f"delivered on {order['delivered_at']}")
        elif order["status"] == "processing":
            reply = (
                f"Order {order['order_id']} is still being prepared in our warehouse and is on track for "
                f"{order['expected_delivery']}. You'll get tracking by email as soon as it ships. "
                "Would you like me to send the tracking link to your email when it's live?"
            )
        else:
            delay = self.store.delay_days(order)
            reply = (
                f"Order {order['order_id']} is {status} with {order['carrier']} "
                f"(tracking {order['tracking_number']}) and is now expected on {order['expected_delivery']}."
            )
            if delay:
                reply += f" It slipped {delay} day(s) because of {order['delay_reason'].lower()}."
                facts.append(f"delayed {delay} days: {order['delay_reason']}")
            threshold = self.store.kb["policies"]["delay_compensation_threshold_days"]
            if delay >= threshold:
                amount = float(self.store.kb["policies"]["delay_compensation_voucher_usd"])
                voucher = self.store.issue_voucher(order["order_id"], amount)
                reply += (
                    f" For the trouble I've added a ${amount:.0f} voucher — code {voucher['code']}."
                )
                facts.append(f"voucher {voucher['code']} for ${amount:.0f}")
                actions.append(
                    Action(
                        type="issue_voucher",
                        detail=f"${amount:.0f} delay compensation",
                        reference=voucher["code"],
                        payload=voucher,
                    )
                )
            reply += " Want me to set up delivery alerts for this shipment?"
        actions.append(
            Action(type="lookup_order", detail=f"Retrieved {order['order_id']}", reference=order["order_id"])
        )
        return Resolution(
            reply=reply,
            facts=facts,
            actions=actions,
            citations=["KB-DEL-01", "KB-DEL-02"],
            solution_note=f"Shared delivery status for {order['order_id']}",
        )

    def _start_return(self, order: dict[str, Any], message: str) -> Resolution:
        window = self.store.return_window_days(order)
        since = self.store.days_since_delivery(order)
        if since is None:
            return Resolution(
                reply=(
                    f"Order {order['order_id']} hasn't been delivered yet, so a return can't start. "
                    "I can cancel it before it ships or set up the return the moment it arrives — which do you prefer?"
                ),
                facts=[f"order {order['order_id']} not delivered", f"return window {window} days"],
                citations=["KB-RET-01"],
                solution_note="Explained return not yet possible",
            )
        if since > window:
            return Resolution(
                reply=(
                    f"Order {order['order_id']} was delivered {since} days ago and this category has a "
                    f"{window}-day return window, so it's outside self-service returns. "
                    "If there's a defect or another special circumstance, tell me and I'll get a specialist to review it."
                ),
                facts=[f"delivered {since} days ago", f"window {window} days", "outside window"],
                citations=["KB-RET-01"],
                solution_note="Return outside window — offered specialist review",
            )
        sku = order["items"][0]["sku"]
        record = self.store.create_return(order["order_id"], sku, reason=message[:180])
        return Resolution(
            reply=(
                f"Done — return {record['rma_id']} is open for {order['items'][0]['name']} on order "
                f"{order['order_id']}, and your prepaid label is on its way to your email "
                f"({record['label_url']}). Drop it off within 7 days and the refund of "
                f"${order['total']:.2f} follows {self.store.kb['policies']['refund_business_days']} business days "
                "after the carrier scan. Would you like a doorstep pickup instead?"
            ),
            facts=[
                f"RMA {record['rma_id']} created",
                f"delivered {since} days ago, within {window}-day window",
                f"refund amount ${order['total']:.2f}",
            ],
            actions=[
                Action(
                    type="create_return",
                    detail=f"Return opened for {order['order_id']}",
                    reference=record["rma_id"],
                    payload=record,
                )
            ],
            citations=["KB-RET-01", "KB-RET-02", "KB-REF-01"],
            solution_note=f"Created return {record['rma_id']}",
        )

    def _refund_update(self, order: dict[str, Any]) -> Resolution:
        existing = self.store.refunds_for_order(order["order_id"])
        eta = self.store.kb["policies"]["refund_business_days"]
        if existing:
            refund = existing[-1]
            return Resolution(
                reply=(
                    f"Refund {refund['refund_id']} for ${refund['amount']:.2f} on order {order['order_id']} is "
                    f"{refund['status'].replace('_', ' ')} and lands back on {order['payment_method']} within "
                    f"{eta} business days. Want me to email you the confirmation?"
                ),
                facts=[f"refund {refund['refund_id']} {refund['status']}", f"eta {eta} business days"],
                citations=["KB-REF-01"],
                solution_note=f"Reported refund status {refund['refund_id']}",
            )
        if order["status"] == "refund_pending":
            return Resolution(
                reply=(
                    f"Order {order['order_id']} already has a refund of ${order['total']:.2f} in progress to "
                    f"{order['payment_method']}; it typically posts within {eta} business days of the return scan. "
                    "Shall I email you an update when it's released?"
                ),
                facts=[f"order {order['order_id']} refund_pending", f"eta {eta} business days"],
                citations=["KB-REF-01"],
                solution_note="Reported in-flight refund",
            )
        record = self.store.create_refund(order["order_id"], order["total"], reason="customer request")
        return Resolution(
            reply=(
                f"I've raised refund {record['refund_id']} for ${order['total']:.2f} on order {order['order_id']} "
                f"back to {order['payment_method']} — it posts within {eta} business days once the return is scanned. "
                "Do you also want a return label for the item?"
            ),
            facts=[f"refund {record['refund_id']} created", f"amount ${order['total']:.2f}"],
            actions=[
                Action(
                    type="create_refund",
                    detail=f"Refund requested for {order['order_id']}",
                    reference=record["refund_id"],
                    payload=record,
                )
            ],
            citations=["KB-REF-01"],
            solution_note=f"Created refund {record['refund_id']}",
        )

    def _cancel(self, order: dict[str, Any]) -> Resolution:
        if order["status"] == "processing":
            order["status"] = "cancelled"
            return Resolution(
                reply=(
                    f"Order {order['order_id']} is cancelled and nothing will ship. The authorization on "
                    f"{order['payment_method']} drops off within 5 business days. "
                    "Would you like help reordering anything?"
                ),
                facts=[f"order {order['order_id']} cancelled", "authorization released in 5 business days"],
                actions=[
                    Action(type="cancel_order", detail="Order cancelled", reference=order["order_id"])
                ],
                citations=["KB-PAY-01"],
                solution_note=f"Cancelled {order['order_id']}",
            )
        return Resolution(
            reply=(
                f"Order {order['order_id']} has already {order['status'].replace('_', ' ')}, so it can't be cancelled. "
                "I can set up a free return as soon as it arrives instead — want me to do that?"
            ),
            facts=[f"order {order['order_id']} status {order['status']}", "cancellation not possible"],
            citations=["KB-RET-01"],
            solution_note="Cancellation not possible — offered return",
        )
