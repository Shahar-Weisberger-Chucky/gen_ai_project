import os
import re
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ── Position detection ────────────────────────────────────────────────────────

def _keyword_position(text: str) -> str | None:
    """
    Fast keyword scan of candidate-only text → exact DB position string.
    ML is checked before Python to avoid misclassifying ML engineers who use Python.
    """
    t = text.lower()
    if re.search(r"\bml\b", t) or "machine learning" in t or "deep learn" in t or "neural" in t:
        return "ML"
    if "sql" in t or "database" in t:
        return "Sql Dev"
    if "analyst" in t:
        return "Analyst"
    if "python" in t:
        return "Python Dev"
    return None

# ── Scheduling intent detection ───────────────────────────────────────────────
# Two-layer check:
#   1. Keywords — only the strongest unambiguous signals, zero API cost
#   2. LLM — handles everything else: typos, synonyms, contextual intent

_SCHEDULE_KEYWORDS = {"schedule", "interview", "appointment", "book a slot"}

def _has_scheduling_intent_keywords(message: str) -> bool:
    t = message.lower()
    return any(kw in t for kw in _SCHEDULE_KEYWORDS)

# ── Prompting strategy: Role + Instructions + Few-Shot + API param (temp=0.4) ─

SYSTEM_PROMPT = """You are an SMS-based recruiter bot for a tech company that hires for four roles:
ML Engineer, SQL Developer, Data Analyst, and Python Developer.

ROLE:
You manage the recruiting conversation with the candidate turn-by-turn.
Your goal: understand which role they are interested in, gather their background,
answer their questions, and ultimately schedule an interview — or end the conversation
politely if they are not interested.

INSTRUCTIONS:
At each turn you MUST choose exactly one action and compose one short message:

  • continue  — keep the dialogue going (ask follow-up questions, answer questions,
                build rapport). Use when you need more information or the candidate
                has a question.

  • schedule  — propose 2–3 specific interview time slots to the candidate.
                Use when the candidate seems interested and qualified, and the
                Scheduling Advisor provides available slots.

  • end       — close the conversation. Use when:
                  (a) The interview has been confirmed (final booking message), OR
                  (b) The Exit Advisor recommends ending AND the candidate is clearly
                      uninterested or has asked to stop contact.

Priority rules:
  0. If the candidate explicitly wants to schedule an interview:
       a. If Detected Position is known AND Scheduling Advisor has slots → action = schedule
       b. If Detected Position is UNKNOWN →
          action = continue, ask ONLY: "Which role are you applying for?
          (ML Engineer, SQL Developer, Data Analyst, or Python Developer)"
          Do NOT ask about experience, qualifications, or anything else.
  1. If Exit Advisor recommends "end" with high confidence → action = end
  2. If interview was just confirmed (candidate agreed on a slot) → action = end
  3. If Scheduling Advisor recommends "schedule" and slots are available → action = schedule
  4. If Info Advisor has relevant content for a candidate question → action = continue
     (include the answer in your message)
  5. Otherwise → action = continue

Keep messages short and professional (SMS style, aim for ≤ 160 characters).
Do NOT include timestamps, labels, or metadata in the message itself.
When the candidate wants to schedule, do NOT ask about experience or qualifications — just schedule.

RESPONSE FORMAT — use EXACTLY this two-line structure, nothing else:
ACTION: <continue|schedule|end>
MESSAGE: <your message to the candidate>
"""

FEW_SHOT = """
--- Example 1: schedule — explicit request, role already known ---
Detected Position: ML
Candidate: lets schedule for ml position
Scheduling Advisor: schedule — 2024-04-10 09:00, 2024-04-11 14:00, 2024-04-14 10:00
ACTION: schedule
MESSAGE: Sure! Available ML slots: Apr 10 at 9 AM, Apr 11 at 2 PM, or Apr 14 at 10 AM. Which works?

--- Example 2: continue — explicit request but role unknown ---
Detected Position: UNKNOWN
Candidate: i want to schedule an interview
Scheduling Advisor: continue | reason: POSITION_UNKNOWN
ACTION: continue
MESSAGE: Happy to schedule! Which role are you applying for? (ML Engineer, SQL Developer, Data Analyst, or Python Developer)

--- Example 3: schedule — role just clarified ---
Detected Position: ML
Candidate: ml engineer
Scheduling Advisor: schedule — 2024-04-10 09:00, 2024-04-11 14:00, 2024-04-14 10:00
ACTION: schedule
MESSAGE: Great! For ML Engineer: Apr 10 at 9 AM, Apr 11 at 2 PM, or Apr 14 at 10 AM. Which works?

--- Example 4: continue (info answer) ---
Detected Position: Analyst
Candidate: What tools does the team use?
Info Advisor: retrieve — We use SQL, Python, and Power BI for analytics.
ACTION: continue
MESSAGE: The analytics team works with SQL, Python, and Power BI. Any other questions?

--- Example 5: end (disinterest) ---
Detected Position: UNKNOWN
Candidate: Please remove me from your list.
Exit Advisor: end (high confidence) — candidate asked to be removed.
ACTION: end
MESSAGE: No worries — I appreciate the update. Best of luck in your search!

--- Example 6: end (booking confirmed) ---
Detected Position: Sql Dev
Candidate: Tuesday at 10 AM works for me.
ACTION: end
MESSAGE: Perfect! Your SQL Dev interview is confirmed for Tuesday at 10 AM. You'll receive a calendar invite.

--- Example 7: continue (general conversation) ---
Detected Position: ML
Candidate: I've been working in machine learning for about two years.
Scheduling Advisor: continue — candidate has not expressed scheduling intent.
ACTION: continue
MESSAGE: Great background! What kind of ML projects have you worked on?
"""


class MainAgent:
    def __init__(self, exit_advisor, scheduling_advisor, info_advisor):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.exit_advisor = exit_advisor
        self.scheduling_advisor = scheduling_advisor
        self.info_advisor = info_advisor
        self.conversation_history: list[dict] = []
        self.candidate_position: str | None = None

    def process_turn(
        self, candidate_message: str, conversation_time: Optional[str] = None
    ) -> tuple[str, str]:
        """
        Handle one candidate message. Detects/updates position, consults all three
        advisors, then asks the LLM to decide the action and write the response.
        Returns (bot_message, action) — action is 'continue'|'schedule'|'end'.
        """
        self.conversation_history.append({"role": "user", "content": candidate_message})

        # Update position state from candidate messages only
        detected = self._detect_position()
        if detected is not None:
            self.candidate_position = detected

        # ── Hard rule: scheduling intent detected ─────────────────────────────
        # Bypass the LLM action decision entirely — the LLM has a training bias
        # toward gathering more info, and this is a business rule not a judgment call.
        if self._has_scheduling_intent(candidate_message):
            if not self.candidate_position:
                message = (
                    "Happy to schedule! Which role are you applying for? "
                    "(ML Engineer, SQL Developer, Data Analyst, or Python Developer)"
                )
                self.conversation_history.append({"role": "assistant", "content": message})
                return message, "continue"

            # Position known — get slots and force schedule
            sched_rec = self.scheduling_advisor.evaluate(
                self._format_history(), conversation_time, self.candidate_position
            )
            slots = sched_rec.get("slots", "none")
            if slots and slots != "none":
                message = (
                    f"Sure! Here are available {self.candidate_position} interview slots: "
                    f"{slots}. Which works for you?"
                )
                self.conversation_history.append({"role": "assistant", "content": message})
                return message, "schedule"
            else:
                message = "No slots are available right now. I'll follow up shortly with options."
                self.conversation_history.append({"role": "assistant", "content": message})
                return message, "continue"
        # ─────────────────────────────────────────────────────────────────────

        history_text = self._format_history()

        exit_rec = self.exit_advisor.evaluate(history_text)
        sched_rec = self.scheduling_advisor.evaluate(
            history_text, conversation_time, self.candidate_position
        )
        info_rec = self.info_advisor.evaluate(history_text, candidate_message)

        position_label = self.candidate_position or "UNKNOWN"
        advisor_block = (
            f"Detected Position: {position_label}\n"
            f"Exit Advisor    → {exit_rec['recommendation']} "
            f"({exit_rec.get('confidence','?')} confidence): {exit_rec.get('reason','')}\n"
            f"Scheduling Adv  → {sched_rec['recommendation']} | "
            f"slots: {sched_rec.get('slots', 'none')} | reason: {sched_rec.get('reason','')}\n"
            f"Info Advisor    → {info_rec['recommendation']} | "
            f"content: {info_rec.get('content', 'N/A')}"
        )

        prompt = (
            f"{FEW_SHOT}\n\n"
            f"--- Current Conversation ---\n{history_text}\n\n"
            f"--- Advisor Recommendations ---\n{advisor_block}\n\n"
            "Now produce the next recruiter turn in the required format."
        )

        response = self.llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])

        action, message = self._parse(response.content)
        self.conversation_history.append({"role": "assistant", "content": message})
        return message, action

    def decide_action_only(self, history_text: str) -> str:
        """
        Lightweight version used by the eval notebook — just predicts the action
        without running the advisors, to keep evaluation costs low.
        """
        prompt = (
            f"{FEW_SHOT}\n\n"
            f"--- Conversation ---\n{history_text}\n\n"
            "Based on the conversation above, what action should the recruiter take next?\n"
            "Respond ONLY with:\n"
            "ACTION: <continue|schedule|end>\n"
            "MESSAGE: (evaluation only — skip)"
        )
        response = self.llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        action, _ = self._parse(response.content)
        return action

    def reset(self):
        self.conversation_history = []
        self.candidate_position = None

    def _has_scheduling_intent(self, message: str) -> bool:
        """
        Layer 1 — keywords: instant True for the strongest unambiguous signals.
        Layer 2 — LLM: handles typos, synonyms, and contextual intent that
                  keywords will never reliably catch.
        """
        if _has_scheduling_intent_keywords(message):
            return True
        response = self.llm.invoke([HumanMessage(content=(
            "A job candidate sent this message during a recruiting conversation.\n"
            "Does the candidate want to schedule an interview or meeting?\n"
            f'Message: "{message}"\n'
            "Answer with only YES or NO."
        ))])
        return response.content.strip().upper().startswith("YES")

    def _detect_position(self) -> str | None:
        """
        Scan only the candidate's messages for the target role.
        Tries keyword matching first (free). Falls back to a focused LLM call
        if keywords are ambiguous. Never overwrites a confirmed position with None.
        """
        candidate_text = " ".join(
            msg["content"]
            for msg in self.conversation_history
            if msg["role"] == "user"
        )

        # Fast path: keyword scan
        position = _keyword_position(candidate_text)
        if position:
            return position

        # If there's not much text yet, don't bother with the LLM call
        if len(candidate_text.split()) < 5:
            return None

        # Fallback: focused LLM call (temperature=0 for determinism)
        response = self.llm.invoke(
            [HumanMessage(content=(
                "A job candidate sent these messages during a recruiting conversation.\n"
                "Which role are they most interested in?\n"
                "Answer with EXACTLY one of: Python Dev, Sql Dev, Analyst, ML, UNKNOWN\n"
                "No explanation — just the role name.\n\n"
                f"Candidate messages:\n{candidate_text}"
            ))],
        )
        result = response.content.strip()
        valid = {"Python Dev", "Sql Dev", "Analyst", "ML"}
        return result if result in valid else None

    def _format_history(self) -> str:
        lines = []
        for msg in self.conversation_history:
            role = "Recruiter" if msg["role"] == "assistant" else "Candidate"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    @staticmethod
    def _parse(text: str) -> tuple[str, str]:
        action = "continue"
        message = text.strip()
        for line in text.strip().splitlines():
            key, _, val = line.partition(":")
            key = key.strip().upper()
            val = val.strip()
            if key == "ACTION" and val.lower() in ("continue", "schedule", "end"):
                action = val.lower()
            elif key == "MESSAGE" and val and val.lower() != "(evaluation only — skip)":
                message = val
        return action, message
