"""
Streamlit chat UI for the TechRecruit Company Recruiting Bot.

Run with:
    streamlit run streamlit_app/streamlit_main.py
"""
import sys
import os
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

# On Streamlit Cloud there's no .env — inject secrets into env so the rest of
# the code (which uses os.getenv) works without any changes.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

from app.main import create_agent

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TechRecruit – Company Recruiting Bot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { max-width: 720px; padding-top: 2rem; }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🤖 TechRecruit – Company Recruiting Bot")
st.caption("Hi! I'm a recruiting assistant for our company. I'm here to help you through the hiring process.")
st.divider()

# ── Action badge helper ───────────────────────────────────────────────────────
ACTION_BADGE = {
    "continue": "🟡 CONTINUE",
    "schedule": "🟢 SCHEDULE",
    "end":      "🔴 END",
}

# ── Slot helpers ──────────────────────────────────────────────────────────────

def _extract_slots(text: str) -> list[str]:
    """Pull all YYYY-MM-DD HH:MM timestamps out of a bot message."""
    return re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", text)

def _format_slot(slot_str: str) -> str:
    """'2024-01-09 10:00'  →  'Wed, Jan 9 at 10:00 AM'"""
    try:
        dt = datetime.strptime(slot_str.strip(), "%Y-%m-%d %H:%M")
        return dt.strftime("%a, %b %-d at %I:%M %p")
    except Exception:
        return slot_str

# ── Session state initialisation ─────────────────────────────────────────────
if "agent" not in st.session_state:
    with st.spinner("Initialising agents…"):
        st.session_state.agent = create_agent()

    opening = (
        "Hi! I'm a recruiting assistant for our company. "
        "I'm here to help you through the hiring process. "
        "Could you tell me a bit about yourself and the role you're interested in?"
    )
    st.session_state.messages = [
        {"role": "assistant", "content": opening, "action": "continue"}
    ]
    st.session_state.agent.conversation_history.append(
        {"role": "assistant", "content": opening}
    )
    st.session_state.ended = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Conversation Info")
    turn_count = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.metric("Candidate turns", turn_count)

    last_action = "—"
    for m in reversed(st.session_state.messages):
        if m["role"] == "assistant" and "action" in m:
            last_action = m["action"].upper()
            break
    st.metric("Last bot action", last_action)

    st.divider()
    st.markdown("**Action legend**")
    st.markdown("🟡 CONTINUE — gathering info / answering")
    st.markdown("🟢 SCHEDULE — proposing interview slots")
    st.markdown("🔴 END — conversation closed")
    st.divider()
    if st.button("🔄 New Conversation"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "action" in msg:
            st.caption(ACTION_BADGE.get(msg["action"], msg["action"].upper()))

# ── Two-phase pattern: process pending bot reply ──────────────────────────────
if "pending_input" in st.session_state and not st.session_state.ended:
    pending = st.session_state.pop("pending_input")
    with st.spinner("Bot is thinking…"):
        conversation_time = datetime.now().isoformat()
        response, action = st.session_state.agent.process_turn(pending, conversation_time)
    st.session_state.messages.append(
        {"role": "assistant", "content": response, "action": action}
    )
    if action == "end":
        st.session_state.ended = True
    st.rerun()

# ── Slot selection UI ─────────────────────────────────────────────────────────
# Show clickable slot buttons when the last bot message is a schedule offer.
last_msg = st.session_state.messages[-1] if st.session_state.messages else None
show_slot_ui = (
    not st.session_state.ended
    and last_msg is not None
    and last_msg.get("action") == "schedule"
    and "pending_input" not in st.session_state
)

if show_slot_ui:
    slots = _extract_slots(last_msg["content"])

    if slots:
        if "confirming_slot" not in st.session_state:
            st.markdown("##### Choose a slot:")
            cols = st.columns(len(slots))
            for i, slot in enumerate(slots):
                if cols[i].button(
                    f"📅 {_format_slot(slot)}",
                    key=f"slot_{slot}",
                    use_container_width=True,
                ):
                    st.session_state.confirming_slot = slot
                    st.rerun()
        else:
            slot = st.session_state.confirming_slot
            st.info(
                f"**Confirm your interview for {_format_slot(slot)}?**\n\n"
                "We'll send you a calendar invite once confirmed."
            )
            col1, col2 = st.columns(2)
            if col1.button("✅ Yes, confirm!", use_container_width=True):
                msg, action = st.session_state.agent.confirm_slot(slot)
                st.session_state.messages.append(
                    {"role": "assistant", "content": msg, "action": action}
                )
                st.session_state.ended = True
                st.session_state.pop("confirming_slot", None)
                st.rerun()
            if col2.button("↩️ Back to slots", use_container_width=True):
                st.session_state.pop("confirming_slot", None)
                st.rerun()

# ── Chat input ────────────────────────────────────────────────────────────────
if not st.session_state.ended:
    user_input = st.chat_input("Type your message…")
    if user_input:
        st.session_state.pop("confirming_slot", None)
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.pending_input = user_input
        st.rerun()
else:
    st.success("🎉 Interview scheduled! Conversation has ended. Use the sidebar to start a new one.")
