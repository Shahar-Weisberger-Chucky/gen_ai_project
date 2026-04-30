import os
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ── Prompting strategy: Role + Instructions + Few-Shot + API param (temp=0.4) ─

SYSTEM_PROMPT = """You are an SMS-based recruiter bot for a Python Developer position at a tech company.

ROLE:
You manage the recruiting conversation with the candidate turn-by-turn.
Your goal: gather their background, answer their questions, and ultimately schedule
an interview — or end the conversation politely if they are not interested.

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
  1. If Exit Advisor recommends "end" with high confidence → action = end
  2. If interview was just confirmed (candidate agreed on a slot) → action = end
  3. If Scheduling Advisor recommends "schedule" and slots are available → action = schedule
  4. If Info Advisor has relevant content for a candidate question → action = continue
     (include the answer in your message)
  5. Otherwise → action = continue

Keep messages short and professional (SMS style, aim for ≤ 160 characters).
Do NOT include timestamps, labels, or metadata in the message itself.

RESPONSE FORMAT — use EXACTLY this two-line structure, nothing else:
ACTION: <continue|schedule|end>
MESSAGE: <your message to the candidate>
"""

FEW_SHOT = """
--- Example 1: continue (info answer) ---
Candidate: What technologies does the team use?
Info Advisor: retrieve — We use Python 3, FastAPI, PostgreSQL, Docker, and AWS ECS.
ACTION: continue
MESSAGE: We use Python 3 + FastAPI on the backend, PostgreSQL, Docker, and deploy to AWS ECS. Any other questions?

--- Example 2: schedule ---
Candidate: I have five years of Python experience and I'd love to set up a meeting.
Scheduling Advisor: schedule — 2024-04-10 10:00, 2024-04-11 14:00, 2024-04-14 09:00
ACTION: schedule
MESSAGE: Great background! Our manager is free Wed Apr 10 at 10 AM, Thu Apr 11 at 2 PM, or Mon Apr 14 at 9 AM. Which works?

--- Example 3: end (disinterest) ---
Candidate: Please remove me from your list.
Exit Advisor: end (high confidence) — candidate asked to be removed.
ACTION: end
MESSAGE: No worries — I appreciate the update. Best of luck in your search!

--- Example 4: end (booking confirmed) ---
Candidate: Tuesday at 10 AM works for me.
ACTION: end
MESSAGE: Perfect! Your interview is confirmed for Tuesday at 10 AM. You'll receive a calendar invite shortly.

--- Example 5: continue (gather info) ---
Candidate: I've been using Python for about two years.
Scheduling Advisor: continue — too early to schedule.
ACTION: continue
MESSAGE: Good to know! Are you also working with any frameworks like Django or FastAPI?
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

    def process_turn(
        self, candidate_message: str, conversation_time: Optional[str] = None
    ) -> tuple[str, str]:
        """
        Handle one candidate message. Consults all three advisors, then
        asks the LLM to decide the action and write the response.
        Returns (bot_message, action) — action is 'continue'|'schedule'|'end'.
        """
        self.conversation_history.append({"role": "user", "content": candidate_message})
        history_text = self._format_history()

        exit_rec = self.exit_advisor.evaluate(history_text)
        sched_rec = self.scheduling_advisor.evaluate(history_text, conversation_time)
        info_rec = self.info_advisor.evaluate(history_text, candidate_message)

        advisor_block = (
            f"Exit Advisor    → {exit_rec['recommendation']} "
            f"({exit_rec.get('confidence','?')} confidence): {exit_rec.get('reason','')}\n"
            f"Scheduling Adv  → {sched_rec['recommendation']} | "
            f"slots: {sched_rec.get('slots', 'none')}\n"
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
