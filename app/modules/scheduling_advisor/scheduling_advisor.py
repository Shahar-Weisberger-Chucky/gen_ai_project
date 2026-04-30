import os
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
import pyodbc
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

load_dotenv()

# ── SQL Tool ──────────────────────────────────────────────────────────────────

@tool
def get_available_slots(reference_date: str) -> str:
    """
    Pull the next 5 available Python Dev interview slots from the SQL Server calendar.
    Pass reference_date as YYYY-MM-DD — returns slots on or after that date.
    If the DB only has historical data (e.g. demo year), falls back to earliest available.
    Returns a comma-separated list of date/time pairs, or an error string.
    """
    driver = os.getenv("SQL_DRIVER", "{ODBC Driver 17 for SQL Server}")
    server = os.getenv("SQL_SERVER", "localhost")
    database = os.getenv("SQL_DATABASE", "Tech")
    trusted = os.getenv("SQL_TRUSTED_CONNECTION", "yes")
    conn_str = (
        f"DRIVER={driver};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"Trusted_Connection={trusted};"
    )
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT TOP (5) [date], [time]
            FROM   dbo.Schedule
            WHERE  position  = 'Python Dev'
              AND  available = 1
              AND  [date]   >= ?
            ORDER  BY [date], [time]
            """,
            reference_date,
        )
        rows = cursor.fetchall()

        # if nothing came back (DB has only past data), just return whatever's earliest
        if not rows:
            cursor.execute(
                """
                SELECT TOP (5) [date], [time]
                FROM   dbo.Schedule
                WHERE  position  = 'Python Dev'
                  AND  available = 1
                ORDER  BY [date], [time]
                """
            )
            rows = cursor.fetchall()

        conn.close()
        if not rows:
            return "No available slots found."
        return ", ".join(f"{str(r[0])} {str(r[1])[:5]}" for r in rows)
    except Exception as exc:
        return f"DB error: {exc}"


# ── Prompting strategy: Role + Instructions + Few-Shot + API param (temp=0) ──

SYSTEM_PROMPT = """You are the Interview Scheduling Advisor for a Python Developer recruiting chatbot.

ROLE:
Decide whether it is the right time to schedule an interview with this candidate.
If yes, use the get_available_slots tool to check the calendar, then pick the best slots
based on anything the candidate said about their availability.

INSTRUCTIONS:
  • Do NOT recommend scheduling in the very first exchange — gather at least a little info first
  • Recommend scheduling when the candidate appears qualified and interested
  • If the candidate mentioned a preferred day or time, prioritise slots closest to that
  • The conversation timestamp is provided so you can resolve relative dates like "next Friday"
  • Always suggest exactly 2 or 3 specific slots when recommending scheduling
  • If no suitable slots are available, recommend "continue" and explain

RESPONSE FORMAT (use exactly this structure):
RECOMMENDATION: <schedule|continue>
SLOTS: <comma-separated datetime strings like "2024-04-10 10:00, 2024-04-11 14:00", or "none">
REASON: <one sentence>
"""

FEW_SHOT = """
--- Example 1 ---
Candidate said "I can do next Wednesday or Thursday".
Available slots: 2024-04-10 09:00, 2024-04-10 14:00, 2024-04-11 10:00
RECOMMENDATION: schedule
SLOTS: 2024-04-10 14:00, 2024-04-11 10:00
REASON: Candidate is interested and available slots match their stated preference.

--- Example 2 ---
First message of the conversation — candidate just introduced themselves.
RECOMMENDATION: continue
SLOTS: none
REASON: Too early to schedule; need to gather more information first.

--- Example 3 ---
Candidate said "Please remove me from your list."
RECOMMENDATION: continue
SLOTS: none
REASON: Candidate is not interested; scheduling is not appropriate.
"""


class SchedulingAdvisor:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,       # deterministic slot selection
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.agent = create_agent(
            model=self.llm,
            tools=[get_available_slots],
            system_prompt=SYSTEM_PROMPT,
        )

    def evaluate(
        self,
        conversation_history: str,
        conversation_time: Optional[str] = None,
    ) -> dict:
        ref_date = self._reference_date(conversation_time)

        user_input = (
            f"{FEW_SHOT}\n\n"
            f"--- Current Conversation ---\n{conversation_history}\n\n"
            f"Conversation timestamp: {conversation_time or 'unknown'}\n"
            f"Call get_available_slots with reference_date='{ref_date}', then decide "
            "whether to schedule an interview now."
        )

        result = self.agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        return self._parse(result["messages"][-1].content)

    @staticmethod
    def _reference_date(conversation_time: Optional[str]) -> str:
        if conversation_time:
            try:
                dt = datetime.fromisoformat(conversation_time.replace("Z", "+00:00"))
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
        return datetime.now().strftime("%Y-%m-%d")

    @staticmethod
    def _parse(text: str) -> dict:
        result = {"recommendation": "continue", "slots": "none", "reason": ""}
        for line in text.strip().splitlines():
            key, _, val = line.partition(":")
            key = key.strip().upper()
            val = val.strip()
            if key == "RECOMMENDATION" and val.lower() in ("schedule", "continue"):
                result["recommendation"] = val.lower()
            elif key == "SLOTS":
                result["slots"] = val
            elif key == "REASON":
                result["reason"] = val
        return result
