# E-commerce AI Customer Support Agent

A runnable support agent for orders, returns, refunds and delivery tracking, with a chat UI,
mock order/knowledge backends, and human escalation with full handoff context.

## Run

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,llm]"
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000.

The agent works with no API key (deterministic rule-based NLU and templated replies).
Set `OPENAI_API_KEY` (optionally `SUPPORT_AGENT_MODEL`, default `gpt-4o-mini`) to enable the LLM layer,
which handles intent/entity extraction and rewrites replies from the grounded facts the core retrieved.
The LLM never invents order data: it is only given facts the backend returned.

## Architecture

| Module | Role |
| --- | --- |
| `app/nlu.py` | Rule-based intent classification, entity extraction (order ID, tracking, email, ZIP, phone), sentiment |
| `app/knowledge.py` | Keyword retrieval over the knowledge base |
| `app/store.py` | Mock order/customer/return/refund/escalation backends and policy helpers |
| `app/agent.py` | Orchestration: identity verification, intent handlers, workflows, escalation triggers |
| `app/llm.py` | Optional LLM layer (understanding + reply phrasing), degrades to rules |
| `app/main.py` | FastAPI API + static chat console |

### Conversation flow

1. Understand — intent, entities, sentiment (LLM if available, rules otherwise).
2. Check escalation triggers before answering.
3. Resolve the order (by order ID or tracking number) and verify identity before any order detail is shared.
4. Run the workflow: delivery update (with delay compensation voucher), return + RMA/label, refund request/status, cancellation, or knowledge-base answer.
5. Phrase the reply from verified facts only: direct answer, supporting detail, closing question.

### Escalation triggers

Product quality/damage, policy-exception requests, payment disputes or complex refunds,
repeated dissatisfaction across turns, strong anger, and questions outside the knowledge base.
Each escalation creates a ticket carrying the full transcript, sentiment, urgency, order/customer
records and the solutions already attempted (`GET /api/escalations`).

### Identity verification

Order details are only shared after the customer confirms the account email, delivery ZIP, or last 4
digits of the phone number. The pending intent is resumed automatically once verification passes.

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /api/chat` | `{session_id, message}` → reply, understanding, actions, citations, escalation |
| `POST /api/sessions/{id}/reset` | Clear conversation state |
| `GET /api/orders`, `GET /api/orders/{id}` | Mock order data |
| `GET /api/escalations` | Human-agent handoff queue |
| `GET /api/health` | Status and whether the LLM layer is active |

## Demo data

| Order | Scenario |
| --- | --- |
| `ORD-10234` | In transit, 3-day weather delay → triggers compensation voucher (ZIP 600042) |
| `ORD-10235` | Delivered recently, inside the return window → self-service RMA (ZIP 10021) |
| `ORD-10236` | Delivered in June, outside the window → refusal or escalation path (ZIP 94107) |
| `ORD-10237` | Processing → cancellable (ZIP 600042) |
| `ORD-10238` | Refund already pending (ZIP 10021) |

## Tests

```bash
pytest -q
ruff check app tests
```
