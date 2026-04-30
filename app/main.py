"""
Start here.

CLI:
    python -m app.main

Streamlit UI:
    streamlit run streamlit_app/streamlit_main.py

First-time setup:
    1. Run db_Tech.sql in SSMS to create the Tech database
    2. python -m app.modules.embedding.embedding  (builds chroma_db from the PDF)
"""
import os
from dotenv import load_dotenv

load_dotenv()

from app.modules.exit_advisor.exit_advisor import ExitAdvisor
from app.modules.scheduling_advisor.scheduling_advisor import SchedulingAdvisor
from app.modules.info_advisor.info_advisor import InfoAdvisor
from app.modules.main_agent.main_agent import MainAgent


def create_agent() -> MainAgent:
    """Wire up all three advisors and hand them to the Main Agent."""
    return MainAgent(
        exit_advisor=ExitAdvisor(),
        scheduling_advisor=SchedulingAdvisor(),
        info_advisor=InfoAdvisor(),
    )


def run_cli():
    """Simple terminal loop — useful for quick testing without opening Streamlit."""
    print("=" * 50)
    print("  Python Developer Recruiter Bot  (CLI mode)")
    print("  Type 'quit' to exit.")
    print("=" * 50 + "\n")

    agent = create_agent()

    opening = (
        "Hi! Thanks for applying to our Python Developer opening. "
        "Could you share a bit about your Python experience?"
    )
    print(f"Bot: {opening}\n")
    agent.conversation_history.append({"role": "assistant", "content": opening})

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        message, action = agent.process_turn(user_input)
        print(f"\nBot [{action.upper()}]: {message}\n")

        if action == "end":
            print("--- Conversation ended ---")
            break


if __name__ == "__main__":
    run_cli()
