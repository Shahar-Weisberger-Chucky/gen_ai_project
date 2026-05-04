"""
Streamlit chat UI for the Python Developer Recruiter Bot.

Run with:
    streamlit run streamlit_app/streamlit_main.py
"""
import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

# on Streamlit Cloud there's no .env — inject secrets into env so the rest of
# the code (which uses os.getenv) works without any changes
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

# ── Custom CSS: SMS-style chat bubbles ────────────────────────────────────────
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

# ── Sidebar: conversation info ────────────────────────────────────────────────
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

# ── Two-phase pattern: show user message first, process bot reply on next run ─
if "pending_input" in st.session_state and not st.session_state.ended:
    pending = st.session_state.pop("pending_input")
    with st.spinner("Bot is thinking…"):
        response, action = st.session_state.agent.process_turn(pending)
    st.session_state.messages.append(
        {"role": "assistant", "content": response, "action": action}
    )
    if action == "end":
        st.session_state.ended = True
    st.rerun()

# ── Chat input ────────────────────────────────────────────────────────────────
if not st.session_state.ended:
    user_input = st.chat_input("Type your message…")
    if user_input:
        # phase 1: show the user message immediately, defer bot processing to next run
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.pending_input = user_input
        st.rerun()
else:
    st.success("Conversation has ended. Use the sidebar to start a new one.")
