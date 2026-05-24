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
    if re.search(r"\bml\b", t) or "machine learning" in t or "deep learn" in t:
        return "ML"
    if "sql" in t or "database" in t:
        return "Sql Dev"
    if "analyst" in t:
        return "Analyst"
    if "python" in t:
        return "Python Dev"
    return None


@tool
def get_available_slots(reference_date: str, position_hint: str) -> str:
    """
    Pull the 3 nearest available interview slots from the SQL Server calendar.
    reference_date: YYYY-MM-DD — return slots on or after this date.
    position_hint: free-text role from the conversation, e.g. "ML", "machine learning",
                   "python developer", "sql", "analyst". The tool normalises it to the DB value.
    Returns a comma-separated list of up to 3 date/time slots, or an error string.
    """
    position = _normalize_position(position_hint)
    if position is None:
        return (
            "POSITION_UNKNOWN: Could not determine the target role from the conversation. "
            "Ask the candidate which role they are applying for: "
            "ML Engineer, SQL Developer, Data Analyst, or Python Developer."
        )

    driver   = os.getenv("SQL_DRIVER", "{ODBC Driver 17 for SQL Server}")
    server   = os.getenv("SQL_SERVER", "localhost")
    database = os.getenv("SQL_DATABASE", "Tech")
    trusted  = os.getenv("SQL_TRUSTED_CONNECTION", "yes")
    conn_str = (
        f"DRIVER={driver};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"Trusted_Connection={trusted};"
    )
    try:
        conn   = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT TOP (3) [date], [time]
            FROM   dbo.Schedule
            WHERE  position  = ?
              AND  available = 1
              AND  [date]   >= ?
            ORDER  BY [date], [time]
            """,
            position, reference_date,
        )
        rows = cursor.fetchall()

        # Fall back to earliest available if nothing exists on/after reference_date
        if not rows:
            cursor.execute(
                """
                SELECT TOP (3) [date], [time]
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
The company is hiring for the Python Developer role.

ROLE:
Decide whether it is the right time to schedule an interview.
When scheduling, call get_available_slots with a computed reference_date and position_hint.

DATE INFERENCE (critical):
  • The conversation timestamp is your definition of "today".
  • When the candidate mentions a relative time ("next Friday", "a week later", "in 2 weeks"),
    compute the actual YYYY-MM-DD date from that timestamp and use it as reference_date.
  • If previously offered slots are provided, relative references like "a week after" or
    "a week later" mean one week after the FIRST previously-offered slot date — not after today.
    Example: offered slots start on 2024-01-02, candidate says "a week later"
             → reference_date = "2024-01-09"
  • If the candidate gives no time preference, use tomorrow's date (today + 1 day) as reference_date.
  • Compute reference_date yourself — do NOT guess or leave it as a default.

POSITION:
  • If the resolved position is provided, use it directly as position_hint.
  • If not, extract from conversation. If unclear, return POSITION_UNKNOWN.

INSTRUCTIONS:
  • IMMEDIATELY recommend schedule if the candidate explicitly requests an interview/meeting.
  • Suggest exactly 3 specific slots when recommending scheduling.
  • If previously offered slots exist, do NOT return slots that overlap with them.
  • If no slots are available, recommend "continue" and explain.

RESPONSE FORMAT (use exactly this structure):
RECOMMENDATION: <schedule|continue>
SLOTS: <comma-separated datetime strings, or "none">
REASON: <one sentence>
"""

FEW_SHOT = """
--- Example 1: explicit request, no prior slots ---
Conversation timestamp: 2024-04-01
Candidate: "lets schedule for ml position"
→ No time preference → reference_date = "2024-04-02"  (today + 1 day)
→ call get_available_slots(reference_date="2024-04-01", position_hint="ml")
→ slots: 2024-04-02 09:00, 2024-04-02 14:00, 2024-04-03 10:00
RECOMMENDATION: schedule
SLOTS: 2024-04-02 09:00, 2024-04-02 14:00, 2024-04-03 10:00
REASON: Candidate explicitly requested an interview for the ML position.

--- Example 2: candidate wants slots a week later ---
Conversation timestamp: 2024-01-01
Previously offered slots: 2024-01-02 09:00, 2024-01-02 12:00, 2024-01-02 13:00
Candidate: "maybe a week later?"
→ "a week later" relative to first offered slot 2024-01-02 → reference_date = "2024-01-09"
→ call get_available_slots(reference_date="2024-01-09", position_hint="ML")
→ slots: 2024-01-09 10:00, 2024-01-10 14:00, 2024-01-11 09:00
RECOMMENDATION: schedule
SLOTS: 2024-01-09 10:00, 2024-01-10 14:00, 2024-01-11 09:00
REASON: Candidate requested slots approximately one week after the previously offered ones.

--- Example 3: candidate says "next Friday" ---
Conversation timestamp: 2024-04-01 (Monday)
Candidate: "can we do next Friday?"
→ next Friday from 2024-04-01 = 2024-04-05 → reference_date = "2024-04-05"
→ call get_available_slots(reference_date="2024-04-05", position_hint="Python Dev")
→ slots: 2024-04-05 09:00, 2024-04-05 14:00, 2024-04-08 10:00
RECOMMENDATION: schedule
SLOTS: 2024-04-05 09:00, 2024-04-05 14:00, 2024-04-08 10:00
REASON: Candidate requested slots around next Friday.

--- Example 4: role unknown ---
Candidate says "i want to schedule an interview" but no role mentioned.
→ call get_available_slots(reference_date="2024-04-01", position_hint="")
→ tool returns POSITION_UNKNOWN
RECOMMENDATION: continue
SLOTS: none
REASON: POSITION_UNKNOWN

--- Example 5: no scheduling intent ---
Candidate just introduced themselves, no scheduling intent expressed.
RECOMMENDATION: continue
SLOTS: none
REASON: Too early to schedule; candidate has not expressed scheduling intent.

--- Example 6: candidate mentions a future month by name ---
Conversation timestamp: 2026-05-24
Previously offered slots: 2026-05-26 12:00, 2026-05-26 13:00, 2026-05-26 15:00
Candidate: "lets schedule sometime in June please"
→ "in June" / "sometime in June" → use the 1st of that month as reference_date = "2026-06-01"
→ call get_available_slots(reference_date="2026-06-01", position_hint="Python Dev")
→ slots: 2026-06-02 10:00, 2026-06-03 09:00, 2026-06-03 10:00
RECOMMENDATION: schedule
SLOTS: 2026-06-02 10:00, 2026-06-03 09:00, 2026-06-03 10:00
REASON: Candidate requested slots in June; used June 1 as reference_date.
"""


class SchedulingAdvisor:
    def _conn_str(self) -> str:
        return (
            f"DRIVER={os.getenv('SQL_DRIVER', '{ODBC Driver 17 for SQL Server}')};"
            f"SERVER={os.getenv('SQL_SERVER', 'localhost')};"
            f"DATABASE={os.getenv('SQL_DATABASE', 'Tech')};"
            f"Trusted_Connection={os.getenv('SQL_TRUSTED_CONNECTION', 'yes')};"
        )

    def mark_slot_booked(self, slot_datetime: str, position: str) -> None:
        """Mark a confirmed slot as unavailable so it isn't offered to other candidates."""
        try:
            parts = slot_datetime.strip().split(" ", 1)
            date_str = parts[0]
            time_str = (parts[1] if len(parts) > 1 else "09:00") + ":00"
            conn = pyodbc.connect(self._conn_str())
            conn.cursor().execute(
                "UPDATE dbo.Schedule SET available = 0 "
                "WHERE [date] = ? AND [time] = ? AND position = ?",
                date_str, time_str, position,
            )
            conn.commit()
            conn.close()
            print(f"[SchedulingAdvisor] Booked: {slot_datetime} ({position})")
        except Exception as exc:
            print(f"[SchedulingAdvisor] mark_slot_booked failed: {exc}")

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
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
        detected_position: Optional[str] = None,
        last_offered_slots: Optional[str] = None,
    ) -> dict:
        today = conversation_time or datetime.now().isoformat()

        position_note = (
            f"Resolved position: '{detected_position}'. "
            f"Use '{detected_position}' as position_hint. Do NOT re-infer from conversation."
            if detected_position
            else
            "Position not yet resolved — extract from conversation or return POSITION_UNKNOWN."
        )

        slot_note = (
            f"Previously offered slots: {last_offered_slots}. "
            "Candidate wants DIFFERENT slots. Compute reference_date from their time preference "
            "relative to the first offered slot date. Do NOT return overlapping slots."
            if last_offered_slots
            else
            "No slots have been offered yet. Use tomorrow's date (today + 1 day) as default reference_date "
            "if the candidate gives no time preference."
        )

        user_input = (
            f"{FEW_SHOT}\n\n"
            f"--- Current Conversation ---\n{conversation_history}\n\n"
            f"Conversation timestamp (today): {today}\n"
            f"Position: {position_note}\n"
            f"Slots: {slot_note}\n\n"
            "Infer the candidate's time preference from the conversation, compute reference_date, "
            "then call get_available_slots and decide whether to schedule."
        )

        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"recursion_limit": 8},   # prevent infinite loops
        )
        return self._parse(result["messages"][-1].content)

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
