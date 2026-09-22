from __future__ import annotations

import json
import random
import string
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .models import Escalation

DATA_DIR = Path(__file__).parent / "data"


def _reference(prefix: str) -> str:
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"{prefix}-{suffix}"


class Store:
    """In-memory mock of the order, return and escalation backends."""

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        raw = json.loads((data_dir / "orders.json").read_text())
        self.orders: dict[str, dict[str, Any]] = raw["orders"]
        self.customers: dict[str, dict[str, Any]] = raw["customers"]
        self.kb: dict[str, Any] = json.loads((data_dir / "knowledge_base.json").read_text())
        self.returns: dict[str, dict[str, Any]] = {}
        self.refunds: dict[str, dict[str, Any]] = {}
        self.escalations: dict[str, Escalation] = {}

    # --- reads -------------------------------------------------------
    def get_order(self, order_id: str | None) -> dict[str, Any] | None:
        if not order_id:
            return None
        return self.orders.get(order_id.upper())

    def get_customer(self, customer_id: str | None) -> dict[str, Any] | None:
        if not customer_id:
            return None
        return self.customers.get(customer_id)

    def find_order_by_tracking(self, tracking: str) -> dict[str, Any] | None:
        normalized = tracking.replace(" ", "").upper()
        for order in self.orders.values():
            number = (order.get("tracking_number") or "").replace(" ", "").upper()
            if number and number == normalized:
                return order
        return None

    def returns_for_order(self, order_id: str) -> list[dict[str, Any]]:
        return [r for r in self.returns.values() if r["order_id"] == order_id]

    def refunds_for_order(self, order_id: str) -> list[dict[str, Any]]:
        return [r for r in self.refunds.values() if r["order_id"] == order_id]

    # --- policy helpers ----------------------------------------------
    def return_window_days(self, order: dict[str, Any]) -> int:
        policies = self.kb["policies"]
        if any(item.get("category") == "electronics" for item in order["items"]):
            return int(policies["electronics_return_window_days"])
        return int(policies["return_window_days"])

    def days_since_delivery(self, order: dict[str, Any], today: date | None = None) -> int | None:
        if not order.get("delivered_at"):
            return None
        today = today or datetime.now().date()
        delivered = datetime.strptime(order["delivered_at"], "%Y-%m-%d").date()
        return (today - delivered).days

    def delay_days(self, order: dict[str, Any]) -> int:
        original = order.get("original_delivery_estimate")
        expected = order.get("expected_delivery")
        if not original or not expected:
            return 0
        d1 = datetime.strptime(original, "%Y-%m-%d").date()
        d2 = datetime.strptime(expected, "%Y-%m-%d").date()
        return max((d2 - d1).days, 0)

    # --- writes ------------------------------------------------------
    def create_return(self, order_id: str, sku: str | None, reason: str) -> dict[str, Any]:
        rma = _reference("RMA")
        record = {
            "rma_id": rma,
            "order_id": order_id,
            "sku": sku,
            "reason": reason,
            "status": "label_issued",
            "label_url": f"https://labels.example.com/{rma}.pdf",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.returns[rma] = record
        return record

    def create_refund(self, order_id: str, amount: float, reason: str) -> dict[str, Any]:
        ref = _reference("REF")
        record = {
            "refund_id": ref,
            "order_id": order_id,
            "amount": amount,
            "reason": reason,
            "status": "pending_review",
            "eta_business_days": self.kb["policies"]["refund_business_days"],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.refunds[ref] = record
        self.orders[order_id]["status"] = "refund_pending"
        return record

    def issue_voucher(self, order_id: str, amount_usd: float) -> dict[str, Any]:
        code = _reference("VCHR")
        return {"code": code, "order_id": order_id, "amount_usd": amount_usd}

    def create_escalation(self, escalation: Escalation) -> Escalation:
        self.escalations[escalation.ticket_id] = escalation
        return escalation

    def next_ticket_id(self) -> str:
        return _reference("ESC")
