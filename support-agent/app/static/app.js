const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("composer");
const inputEl = document.getElementById("input");
const understandingEl = document.getElementById("understanding");
const actionsEl = document.getElementById("actions");
const escalationsEl = document.getElementById("escalations");
const ordersEl = document.getElementById("orders");
const suggestionsEl = document.getElementById("suggestions");
const badgeEl = document.getElementById("llm-badge");

let sessionId = crypto.randomUUID();
const allActions = [];

const SUGGESTIONS = [
  "Where is my order ORD-10234?",
  "I want to return ORD-10235, my zip is 10021",
  "When will my refund for ORD-10238 arrive?",
  "My espresso machine from ORD-10236 arrived cracked",
  "How long do I have to return electronics?",
];

function addMessage(role, text, meta) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  if (meta && meta.length) {
    const tags = document.createElement("div");
    tags.className = "tags";
    meta.forEach(([label, isEsc]) => {
      const tag = document.createElement("span");
      tag.className = isEsc ? "tag esc" : "tag";
      tag.textContent = label;
      tags.appendChild(tag);
    });
    el.appendChild(tags);
  }
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderUnderstanding(u, verified) {
  const entities = Object.entries(u.entities).filter(([, v]) => v);
  const sentimentClass =
    u.sentiment === "angry" || u.sentiment === "frustrated" ? "pill bad" : "pill good";
  understandingEl.innerHTML = `
    <dt>Intent</dt><dd><span class="pill">${u.intent}</span> <span class="muted">${(u.confidence * 100).toFixed(0)}% · ${u.source}</span></dd>
    <dt>Sentiment</dt><dd><span class="${sentimentClass}">${u.sentiment}</span></dd>
    <dt>Identity</dt><dd><span class="${verified ? "pill good" : "pill bad"}">${verified ? "verified" : "unverified"}</span></dd>
    <dt>Entities</dt><dd>${entities.length ? entities.map(([k, v]) => `${k}: <b>${v}</b>`).join("<br/>") : '<span class="muted">none</span>'}</dd>
  `;
}

function renderActions() {
  if (!allActions.length) {
    actionsEl.innerHTML = '<li class="muted">None yet.</li>';
    return;
  }
  actionsEl.innerHTML = allActions
    .map((a) => `<li><b>${a.type}</b><br/><span class="muted">${a.detail}${a.reference ? ` · ${a.reference}` : ""}</span></li>`)
    .join("");
}

async function refreshEscalations() {
  const res = await fetch("/api/escalations");
  const items = await res.json();
  escalationsEl.innerHTML = items.length
    ? items
        .map(
          (e) =>
            `<li><b>${e.ticket_id}</b> <span class="pill bad">${e.urgency}</span><br/><span class="muted">${e.reason}${e.order_id ? ` · ${e.order_id}` : ""}</span></li>`
        )
        .join("")
    : '<li class="muted">None.</li>';
}

async function send(text) {
  addMessage("customer", text);
  inputEl.value = "";
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: text }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const meta = [];
    data.citations.forEach((c) => meta.push([c, false]));
    if (data.escalated) meta.push([`escalated · ${data.escalation.ticket_id}`, true]);
    addMessage("agent", data.reply, meta);
    renderUnderstanding(data.understanding, data.identity_verified);
    data.actions.forEach((a) => allActions.unshift(a));
    renderActions();
    if (data.escalated) refreshEscalations();
  } catch (err) {
    addMessage("system", `Request failed: ${err.message}`);
  }
}

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (text) send(text);
});

document.getElementById("reset").addEventListener("click", async () => {
  await fetch(`/api/sessions/${sessionId}/reset`, { method: "POST" });
  sessionId = crypto.randomUUID();
  allActions.length = 0;
  messagesEl.innerHTML = "";
  renderActions();
  understandingEl.innerHTML = '<dd class="muted">No turns yet.</dd>';
  greet();
});

function greet() {
  addMessage("agent", "Hi! I can help with order status, delivery tracking, returns, and refunds. What's going on with your order today?");
}

SUGGESTIONS.forEach((s) => {
  const btn = document.createElement("button");
  btn.textContent = s;
  btn.addEventListener("click", () => send(s));
  suggestionsEl.appendChild(btn);
});

(async function init() {
  greet();
  const health = await (await fetch("/api/health")).json();
  badgeEl.textContent = health.llm_enabled ? `LLM: ${health.model}` : "LLM: off (rule-based)";
  badgeEl.className = health.llm_enabled ? "badge on" : "badge";
  const { orders } = await (await fetch("/api/orders")).json();
  ordersEl.innerHTML = orders
    .map((o) => `<li>${o.order_id} · ${o.status}<br/><span class="muted">${o.customer} · eta ${o.expected_delivery}</span></li>`)
    .join("");
  refreshEscalations();
})();
