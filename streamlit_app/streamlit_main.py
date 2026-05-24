"""
Streamlit chat UI for the TechRecruit Company Recruiting Bot.

Run with:
    streamlit run streamlit_app/streamlit_main.py
"""
import sys
import os
import re
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

from app.main import create_agent

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TechRecruit – Python Developer",
    page_icon="🐍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { max-width: 740px; padding-top: 1rem; }
    .stChatMessage { border-radius: 16px; margin-bottom: 0.25rem; }

    .badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-top: 5px;
    }
    .badge-continue { background:#fef9c3; color:#854d0e; border:1px solid #fde047; }
    .badge-schedule { background:#dcfce7; color:#166534; border:1px solid #86efac; }
    .badge-end      { background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; }

    .tips-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 0.8rem 1rem;
        border-radius: 0 10px 10px 0;
        margin: 0.5rem 0 1.2rem;
        font-size: 0.9rem;
        line-height: 1.7;
    }
    .tips-box b { color: #1d4ed8; }

    .slot-hint {
        color: #6b7280;
        font-size: 0.82rem;
        text-align: center;
        margin: 0.3rem 0 0.6rem;
    }

    /* Slot pick buttons and end-screen buttons — light blue, hover darkens */
    div[data-testid="column"] .stButton > button {
        background-color: #dbeafe !important;
        border: 1px solid #93c5fd !important;
        color: #1d4ed8 !important;
        font-weight: 600;
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }
    div[data-testid="column"] .stButton > button:hover {
        background-color: #bfdbfe !important;
        border-color: #3b82f6 !important;
        color: #1e3a8a !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
            color: white; padding: 1.4rem 1.8rem; border-radius: 14px; margin-bottom: 1.2rem;">
    <div style="font-size: 1.7rem; font-weight: 700; margin-bottom: 0.25rem;">
        🐍 Python Developer Recruitment
    </div>
    <div style="opacity: 0.82; font-size: 0.9rem; letter-spacing: 0.02em;">
        TechRecruit &nbsp;·&nbsp; AI Recruiting Assistant &nbsp;·&nbsp; Python Developer Position
    </div>
</div>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
BADGE_HTML = {
    "continue": '<span class="badge badge-continue">● Gathering info</span>',
    "schedule": '<span class="badge badge-schedule">● Scheduling interview</span>',
    "end":      '<span class="badge badge-end">● Conversation closed</span>',
}

# Greeting reflects that the candidate already submitted an application
OPENING_MESSAGE = (
    "Hi! We received your application for our Python Developer position. "
    "I'm your recruiting assistant — could you tell me a bit about your Python experience?"
)

CONVERSATIONS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "conversations")
)

_CONFIRM_KEYWORDS = {"confirmed", "scheduled", "booked", "calendar invite", "look forward"}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_slot(slot_str: str) -> str:
    """Cross-platform slot formatter: '2026-06-03 09:00' → 'Tue, Jun 3 at 09:00 AM'."""
    if not slot_str:
        return "your scheduled time"
    try:
        dt = datetime.strptime(slot_str.strip(), "%Y-%m-%d %H:%M")
        return dt.strftime(f"%a, %b {dt.day} at %I:%M %p")
    except Exception:
        return slot_str

def _is_scheduled_end(message: str) -> bool:
    """True when a conversation ended because an interview was confirmed."""
    m = message.lower()
    return any(kw in m for kw in _CONFIRM_KEYWORDS)

def _is_happy_end() -> bool:
    """True if the most recent assistant message indicates a confirmed interview."""
    for m in reversed(st.session_state.get("messages", [])):
        if m["role"] == "assistant":
            return _is_scheduled_end(m["content"])
    return False

def _save_conversation() -> None:
    """Persist the current conversation to conversations/ as a JSON file.
    If the session was loaded from a file, overwrites that file in place
    so there are no duplicates in the history list.
    """
    msgs = st.session_state.get("messages", [])
    # nothing worth saving if no user messages exist
    if not any(m["role"] == "user" for m in msgs):
        return
    os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
    # overwrite original file if this session was loaded from one
    loaded_from = st.session_state.get("loaded_from_path")
    if loaded_from and os.path.exists(loaded_from):
        filepath = loaded_from
    else:
        filename = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(CONVERSATIONS_DIR, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {"saved_at": datetime.now().isoformat(), "messages": msgs},
                f, indent=2, ensure_ascii=False,
            )
    except Exception as exc:
        st.warning(f"Could not save conversation: {exc}")

def _list_conversations() -> list[dict]:
    """Return saved conversations sorted newest-first (up to 15)."""
    if not os.path.exists(CONVERSATIONS_DIR):
        return []
    files = sorted(
        [f for f in os.listdir(CONVERSATIONS_DIR) if f.endswith(".json")],
        reverse=True,
    )
    result = []
    for fname in files[:15]:
        path = os.path.join(CONVERSATIONS_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            saved_at = data.get("saved_at", "")
            try:
                dt = datetime.fromisoformat(saved_at)
                label = dt.strftime("%b %d, %Y  %H:%M")
            except Exception:
                label = fname
            msg_count = sum(1 for m in data.get("messages", []) if m["role"] == "user")
            result.append({"path": path, "label": label, "msg_count": msg_count, "fname": fname})
        except Exception:
            pass
    return result

def _guess_position(messages: list) -> str | None:
    """Keyword-only position detection from a message list — no LLM cost."""
    text = " ".join(m["content"] for m in messages if m["role"] == "user").lower()
    if re.search(r"\bml\b", text) or "machine learning" in text or "deep learn" in text:
        return "ML"
    if "sql" in text or "database" in text:
        return "Sql Dev"
    if "analyst" in text:
        return "Analyst"
    if "python" in text:
        return "Python Dev"
    return None

def _load_past_conversation(filepath: str) -> None:
    """Restore a saved conversation into session state and rebuild the agent."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    messages = data.get("messages", [])

    if "agent" not in st.session_state:
        with st.spinner("Initialising agent…"):
            st.session_state.agent = create_agent()

    agent = st.session_state.agent
    agent.reset()

    for m in messages:
        if m["role"] in ("user", "assistant"):
            agent.conversation_history.append({"role": m["role"], "content": m["content"]})

    agent.candidate_position = _guess_position(messages)

    for m in reversed(messages):
        if m.get("action"):
            agent.last_action = m["action"]
            break

    st.session_state.messages = messages
    st.session_state.ended = (agent.last_action == "end")
    st.session_state.show_snow = False
    st.session_state.show_confirmation_popup = False
    st.session_state.confirmed_slot_display = ""
    st.session_state.loaded_from_path = filepath  # track so save overwrites, not duplicates


# ── Confirmation popup (requires Streamlit >= 1.36) ───────────────────────────
@st.dialog("Interview Confirmed! 🎉")
def _show_confirmation_popup():
    slot = st.session_state.get("confirmed_slot_display", "your scheduled time")
    st.markdown(f"""
        <div style="text-align:center; padding: 0.5rem 0 1rem;">
            <div style="font-size:3.2rem; margin-bottom:0.8rem;">📅</div>
            <div style="font-size:1.05rem; color:#374151; margin-bottom:0.4rem;">
                Your interview has been scheduled at:
            </div>
            <div style="font-size:1.45rem; font-weight:800; color:#2563eb;
                        background:#eff6ff; padding:0.55rem 1.2rem; border-radius:10px;
                        display:inline-block; margin:0.3rem 0 0.9rem;">
                {slot}
            </div>
            <div style="color:#6b7280; font-size:0.88rem;">
                A calendar invite will be sent to you shortly. Good luck!
            </div>
        </div>
    """, unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("Got it! 🎉", use_container_width=True, type="primary"):
            st.rerun()


# ── Session state init ─────────────────────────────────────────────────────────
if "agent" not in st.session_state:
    try:
        with st.spinner("Initialising agents…"):
            st.session_state.agent = create_agent()
    except Exception as exc:
        st.error(f"Failed to initialise the recruiting agent: {exc}")
        st.info("Check that your OPENAI_API_KEY is set in the .env file and try refreshing.")
        st.stop()

    st.session_state.messages = [
        {"role": "assistant", "content": OPENING_MESSAGE, "action": "continue"}
    ]
    st.session_state.agent.conversation_history.append(
        {"role": "assistant", "content": OPENING_MESSAGE}
    )
    st.session_state.ended = False
    st.session_state.show_snow = False
    st.session_state.show_confirmation_popup = False
    st.session_state.confirmed_slot_display = ""
    st.session_state.loaded_from_path = None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Conversation Status")

    turn_count = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.metric("Your messages", turn_count)

    last_action = "—"
    for m in reversed(st.session_state.messages):
        if m["role"] == "assistant" and "action" in m:
            last_action = m["action"]
            break
    status_color = {"continue": "#f59e0b", "schedule": "#22c55e", "end": "#ef4444"}.get(last_action, "#999")
    st.markdown(
        f"**Bot status:** <span style='color:{status_color}; font-weight:700;'>{last_action.upper()}</span>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("**Status legend**")
    st.markdown("🟡 **CONTINUE** — gathering info / answering")
    st.markdown("🟢 **SCHEDULE** — proposing interview slots")
    st.markdown("🔴 **END** — conversation finished")

    st.divider()
    st.markdown("**About this role**")
    st.markdown(
        "We're hiring for a **Python Developer** position. "
        "Ask about the tech stack, work model, requirements, or schedule your interview!"
    )

    st.divider()
    if st.button("🔄 New Conversation", use_container_width=True):
        _save_conversation()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # ── Past conversations ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Past Conversations**")
    past_convs = _list_conversations()
    if not past_convs:
        st.caption("No saved conversations yet.")
    else:
        for conv_info in past_convs:
            n = conv_info["msg_count"]
            header = f"📋 {conv_info['label']}  ·  {n} msg{'s' if n != 1 else ''}"
            with st.expander(header):
                if st.button(
                    "Load & Continue",
                    key=f"load_{conv_info['fname']}",
                    use_container_width=True,
                ):
                    _load_past_conversation(conv_info["path"])
                    st.rerun()


# ── Tips (before first user message only) ─────────────────────────────────────
if not any(m["role"] == "user" for m in st.session_state.messages) and not st.session_state.ended:
    st.markdown("""
<div class="tips-box">
    <b>What you can ask me:</b><br>
    &nbsp;&nbsp;• What's the tech stack or salary range?<br>
    &nbsp;&nbsp;• Is the role remote or hybrid?<br>
    &nbsp;&nbsp;• What Python experience is required?<br>
    &nbsp;&nbsp;• Ready to schedule an interview!
</div>
""", unsafe_allow_html=True)

# ── Render chat history ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "action" in msg:
            badge = BADGE_HTML.get(msg["action"], "")
            if badge:
                st.markdown(badge, unsafe_allow_html=True)

# ── Two-phase: process pending bot reply ───────────────────────────────────────
if "pending_input" in st.session_state and not st.session_state.ended:
    pending = st.session_state.pop("pending_input")
    try:
        with st.spinner("Assistant is thinking…"):
            response, action = st.session_state.agent.process_turn(
                pending, datetime.now().isoformat()
            )
    except Exception as exc:
        st.error(f"Something went wrong — please try sending your message again. ({type(exc).__name__})")
        st.session_state.messages.pop()          # remove the unprocessed user message
        st.rerun()
    else:
        st.session_state.messages.append(
            {"role": "assistant", "content": response, "action": action}
        )
        if action == "end":
            st.session_state.ended = True
            happy = _is_scheduled_end(response)
            st.session_state.show_snow = happy
            if happy:
                raw_slot = st.session_state.agent.last_confirmed_slot
                st.session_state.confirmed_slot_display = (
                    _format_slot(raw_slot) if raw_slot else "your scheduled time"
                )
                st.session_state.show_confirmation_popup = True
        st.rerun()

# ── Snow animation on happy end (once) ────────────────────────────────────────
if st.session_state.get("show_snow"):
    st.snow()
    st.session_state.show_snow = False

# ── Confirmation popup (once, on happy end) ────────────────────────────────────
if st.session_state.get("show_confirmation_popup"):
    st.session_state.show_confirmation_popup = False
    _show_confirmation_popup()

# ── Slot selection UI ──────────────────────────────────────────────────────────
# Read raw slots from the agent (LLM rewrites datetimes as "May 28 at 9 AM" in the
# message text, so _extract_slots on the message would find nothing).
last_msg = st.session_state.messages[-1] if st.session_state.messages else None
_raw = st.session_state.agent.last_offered_slots or ""
_agent_slots = [s.strip() for s in _raw.split(",") if s.strip()] if _raw else []

show_slot_ui = (
    not st.session_state.ended
    and last_msg is not None
    and last_msg.get("action") == "schedule"
    and "pending_input" not in st.session_state
    and bool(_agent_slots)
)

if show_slot_ui:
    slots = _agent_slots
    if slots:
        st.markdown("##### 📅 Pick a slot:")
        cols = st.columns(len(slots))
        for i, slot in enumerate(slots):
            if cols[i].button(
                _format_slot(slot),
                key=f"slot_{slot}",
                use_container_width=True,
            ):
                msg, action = st.session_state.agent.confirm_slot(slot)
                st.session_state.messages.append(
                    {"role": "user", "content": f"I'll take the {slot} slot."}
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": msg, "action": action}
                )
                st.session_state.ended = True
                st.session_state.show_snow = True
                raw_slot = st.session_state.agent.last_confirmed_slot or slot
                st.session_state.confirmed_slot_display = _format_slot(raw_slot)
                st.session_state.show_confirmation_popup = True
                st.rerun()
        st.markdown('<div class="slot-hint">— or describe your preference in the chat below —</div>', unsafe_allow_html=True)

# ── Chat input ─────────────────────────────────────────────────────────────────
if not st.session_state.ended:
    user_input = st.chat_input("Type your message…")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.pending_input = user_input
        st.rerun()

# ── Ended screen ───────────────────────────────────────────────────────────────
else:
    if _is_happy_end():
        slot_display = st.session_state.get("confirmed_slot_display", "")
        slot_line = (
            f"Confirmed for <strong>{slot_display}</strong>."
            if slot_display
            else "We'll be in touch with the details."
        )
        st.markdown(f"""
<div style="background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
            color: white; padding: 1.8rem; border-radius: 14px;
            text-align: center; margin: 1rem 0;">
    <div style="font-size: 2.2rem; margin-bottom: 0.4rem;">📅</div>
    <div style="font-size: 1.4rem; font-weight: 700; margin-bottom: 0.3rem;">
        Your interview has been scheduled!
    </div>
    <div style="opacity: 0.88; font-size: 0.92rem;">{slot_line}</div>
</div>
""", unsafe_allow_html=True)
    else:
        st.info("Conversation ended.")

    col1, col2 = st.columns(2)

    if col1.button("💬 Continue Conversation", use_container_width=True):
        continuation_msg = "Of course! Happy to keep chatting. What else can I help you with?"
        st.session_state.messages.append(
            {"role": "assistant", "content": continuation_msg, "action": "continue"}
        )
        st.session_state.agent.conversation_history.append(
            {"role": "assistant", "content": continuation_msg}
        )
        # Reset agent state so it behaves as mid-conversation, not post-end.
        st.session_state.agent.last_action = "continue"
        st.session_state.agent.last_offered_slots = None   # clear old slot buttons
        st.session_state.agent.in_continuation = True      # suppress ExitAdvisor for 1 turn
        st.session_state.ended = False
        st.rerun()

    if col2.button("🔄 New Conversation", use_container_width=True):
        _save_conversation()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ── Always scroll to bottom so new messages stay in view ──────────────────────
# Note: st.markdown scripts run directly in the Streamlit page DOM,
# so use `document` (not window.parent.document which is for component iframes).
st.markdown("""
<script>
    setTimeout(function () {
        const main = document.querySelector('section[data-testid="stMain"]')
                  || document.querySelector('.main');
        if (main) main.scrollTop = main.scrollHeight;
    }, 150);
</script>
""", unsafe_allow_html=True)
