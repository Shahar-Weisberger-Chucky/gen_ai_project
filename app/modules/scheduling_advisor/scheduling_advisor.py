import os
import re
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
import pyodbc
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

load_dotenv()

# ── SQL Tool ──────────────────────────────────────────────────────────────────


def _normalize_position(text: str) -> str | None:
    """Map free-text role mention to the exact DB position string."""
    t = text.lower()
    if "python" in t:
        return "Python Dev"
    if "sql" in t or "database" in t:
        return "Sql Dev"
    if re.search(r"\bml\b", t) or "machine learning" in t or "deep learn" in t:
        return "ML"
    if "analyst" in t or "data" in t:
        return "Analyst"
    return None


@tool
def get_available_slots(reference_date: str, position_hint: str) -> str:
    """
    Pull the next 5 available interview slots from the SQL Server calendar.
    reference_date: YYYY-MM-DD
    position_hint: free-text role from the conversation, e.g. "ML", "machine learning",
                   "python developer", "sql", "analyst". The tool maps it to the DB value.
    Returns a comma-separated list of date/time slots, or an error/clarification string.
    """
    position = _normalize_position(position_hint)
    if position is None:
        return (
            "POSITION_UNKNOWN: Could not determine the target role from the conversation. "
            "Ask the candidate which role they are applying for: "
            "ML Engineer, SQL Developer, Data Analyst, or Python Developer."
        )

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
            WHERE  position  = ?
              AND  available = 1
              AND  [date]   >= ?
            ORDER  BY [date], [time]
            """,
            position, reference_date,
        )
        rows = cursor.fetchall()

        if not rows:
            cursor.execute(
                """
                SELECT TOP (5) [date], [time]
                FROM   dbo.Schedule
                WHERE  position  = ?
                  AND  available = 1
                ORDER  BY [date], [time]
                """,
                position,
            )
            rows = cursor.fetchall()

        conn.close()
        if not rows:
            return "No available slots found."
        return ", ".join(f"{str(r[0])} {str(r[1])[:5]}" for r in rows)
    except Exception as exc:
        return f"DB error: {exc}"


# ── Prompting strategy: Role + Instructions + Few-Shot + API param (temp=0) ──

SYSTEM_PROMPT = """You are the Interview Scheduling Advisor for a tech company recruiting chatbot.
The company hires for four roles: ML Engineer, SQL Developer, Data Analyst, Python Developer.

ROLE:
Decide whether it is the right time to schedule an interview.
When scheduling, call get_available_slots with reference_date and position_hint extracted
from the conversation. The tool handles mapping the role to the DB — pass the role as the
candidate described it (e.g. "ml", "machine learning", "python", "sql", "analyst").

INSTRUCTIONS:
  • IMMEDIATELY recommend schedule if the candidate explicitly requests an interview/meeting.
    Do NOT ask about experience or qualifications in that case.
  • If the candidate wants to schedule but the role is not clear from the conversation,
    recommend "continue" with REASON: POSITION_UNKNOWN so the main agent can ask.
  • If the tool returns POSITION_UNKNOWN, recommend "continue" with that reason.
  • Recommend scheduling when the candidate appears interested, even with minimal context.
  • Suggest exactly 2 or 3 specific slots when recommending scheduling.
  • If no slots are available, recommend "continue" and explain.

RESPONSE FORMAT (use exactly this structure):
RECOMMENDATION: <schedule|continue>
SLOTS: <comma-separated datetime strings, or "none">
REASON: <one sentence>
"""

FEW_SHOT = """
--- Example 1: explicit request, role clear ---
Conversation: Candidate wants ML position and says "lets schedule for ml position"
→ call get_available_slots(reference_date="2024-04-01", position_hint="ml")
→ slots returned: 2024-04-10 09:00, 2024-04-11 14:00, 2024-04-14 10:00
RECOMMENDATION: schedule
SLOTS: 2024-04-10 09:00, 2024-04-11 14:00, 2024-04-14 10:00
REASON: Candidate explicitly requested an interview for the ML position.

--- Example 2: explicit request, role unknown ---
Conversation: Candidate says "i want to schedule an interview" but no role mentioned yet.
→ call get_available_slots(reference_date="2024-04-01", position_hint="")
→ tool returns POSITION_UNKNOWN
RECOMMENDATION: continue
SLOTS: none
REASON: POSITION_UNKNOWN

--- Example 3: preference given, role clear ---
Candidate said "I can do next Wednesday or Thursday" for a Python Dev role.
→ slots: 2024-04-10 14:00, 2024-04-11 10:00
RECOMMENDATION: schedule
SLOTS: 2024-04-10 14:00, 2024-04-11 10:00
REASON: Candidate is interested and slots match their stated preference.

--- Example 4: first message only ---
Candidate just introduced themselves, no scheduling intent expressed.
RECOMMENDATION: continue
SLOTS: none
REASON: Too early to schedule; candidate has not expressed scheduling intent.

--- Example 5: disinterest ---
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
