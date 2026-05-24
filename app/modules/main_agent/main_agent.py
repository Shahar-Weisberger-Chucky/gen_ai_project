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
    Currently only Python Developer is active; add more roles here when scaling up.
    """
    if "python" in text.lower():
        return "Python Dev"
    return None


# ── Scheduling intent detection ───────────────────────────────────────────────
# Two-layer check:
#   1. Keywords — only the strongest unambiguous signals, zero API cost
#   2. LLM — handles everything else: typos, synonyms, contextual intent

_SCHEDULE_KEYWORDS = {"schedule", "interview", "appointment", "book a slot", "reschedule", "rescheduling"}

def _has_scheduling_intent_keywords(message: str) -> bool:
    t = message.lower()
    return any(kw in t for kw in _SCHEDULE_KEYWORDS)


# ── Prompting strategy: Role + Instructions + Few-Shot + API param (temp=0.4) ─

SYSTEM_PROMPT = """You are an SMS-based recruiter bot for a tech company hiring for the Python Developer role.

ROLE:
You manage the recruiting conversation with the candidate turn-by-turn.
Your goal: gather the candidate's Python background, answer their questions, and
ultimately schedule an interview — or end the conversation politely if they are not interested.

INSTRUCTIONS:
At each turn you MUST choose exactly one action and compose one short message:

  • continue  — keep the dialogue going (ask follow-up questions, answer questions,
                build rapport). Use when you need more information or the candidate
                has a question.

  • schedule  — propose 2–3 specific interview time slots to the candidate.
                Use when the candidate seems interested and the Scheduling Advisor
                provides available slots.

  • end       — close the conversation. Use when:
                  (a) The interview has been confirmed (final booking message), OR
                  (b) The Exit Advisor recommends ending AND the candidate is clearly
                      uninterested or has asked to stop contact.

Priority rules:
  0. If the candidate explicitly wants to schedule an interview and the
     Scheduling Advisor has slots → action = schedule
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
--- Example 1: schedule — explicit request ---
Detected Position: Python Dev
Candidate: lets schedule an interview
Scheduling Advisor: schedule — 2024-04-10 09:00, 2024-04-11 14:00, 2024-04-14 10:00
ACTION: schedule
MESSAGE: Sure! Available Python Dev slots: Apr 10 at 9 AM, Apr 11 at 2 PM, or Apr 14 at 10 AM. Which works?

--- Example 2: schedule — candidate wants different time ---
Detected Position: Python Dev
Context: We already offered slots on Jan 2. Candidate wants a week later.
Scheduling Advisor: schedule — 2024-01-09 10:00, 2024-01-10 14:00, 2024-01-11 09:00
ACTION: schedule
MESSAGE: Sure! Here are Python Dev slots around that time: Jan 9 at 10 AM, Jan 10 at 2 PM, or Jan 11 at 9 AM. Which works?

--- Example 3: end — candidate confirmed a slot ---
Detected Position: Python Dev
Candidate: ok 2024-01-02 09:00 works for me
ACTION: end
MESSAGE: Perfect! Your Python Dev interview is confirmed for Jan 2 at 9 AM. You'll receive a calendar invite shortly.

--- Example 4: continue (info answer) ---
Detected Position: Python Dev
Candidate: What tools does the team use?
Info Advisor: retrieve — We use Django, Flask, and FastAPI for backend services.
ACTION: continue
MESSAGE: The team works with Django, Flask, and FastAPI. Any other questions?

--- Example 5: end (disinterest) ---
Detected Position: Python Dev
Candidate: Please remove me from your list.
Exit Advisor: end (high confidence) — candidate asked to be removed.
ACTION: end
MESSAGE: No worries — I appreciate the update. Best of luck in your search!

--- Example 6: continue (general conversation) ---
Detected Position: Python Dev
Candidate: I've been working with Python for about two years, mostly Django.
Scheduling Advisor: continue — candidate has not expressed scheduling intent.
ACTION: continue
MESSAGE: Great background! What kind of Python projects have you worked on?
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
        self.candidate_position: str = "Python Dev"  # single-role mode; extend _keyword_position to add more
        self.last_action: str = "continue"
        self.last_offered_slots: str | None = None
        self.last_confirmed_slot: str | None = None
        self.in_continuation: bool = False  # set True by UI after "Continue Conversation"

    def process_turn(
        self, candidate_message: str, conversation_time: Optional[str] = None
    ) -> tuple[str, str]:
        """
        Handle one candidate message. Detects/updates position, consults advisors,
        then decides action. Returns (bot_message, action).
        """
        self.conversation_history.append({"role": "user", "content": candidate_message})

        detected = self._detect_position()
        if detected is not None:
            self.candidate_position = detected

        # ── State: candidate is responding to slots we already offered ─────────
        # Skip the hard scheduling rule — they are in a slot negotiation dialogue.
        if self.last_action == "schedule":
            return self._handle_slot_response(candidate_message, conversation_time)

        # ── Hard rule: fresh scheduling intent ────────────────────────────────
        # Bypass LLM action decision — the LLM has a training bias toward
        # gathering more info, and "should I schedule?" is a business rule.
        if self._has_scheduling_intent(candidate_message):
            sched_rec = self.scheduling_advisor.evaluate(
                self._format_history(), conversation_time, self.candidate_position
            )
            slots = sched_rec.get("slots", "none")
            if slots and slots != "none":
                message = (
                    f"Sure! Here are available {self.candidate_position} interview slots: "
                    f"{slots}. Which works for you?"
                )
                return self._record_and_return("schedule", message, slots=slots)
            else:
                message = "No slots are available right now. I'll follow up shortly with options."
                return self._record_and_return("continue", message)

        # ── Normal LLM flow ───────────────────────────────────────────────────
        return self._llm_turn(candidate_message, conversation_time)

    def _handle_slot_response(
        self, candidate_message: str, conversation_time: Optional[str]
    ) -> tuple[str, str]:
        """
        Called when last_action == 'schedule'. The candidate is responding to
        slots we offered — either confirming one, requesting different times,
        or asking a question. Never re-runs the hard scheduling rule here.
        """
        confirmed_slot = self._is_slot_confirmed(candidate_message)
        if confirmed_slot:
            # Candidate may have confirmed a slot AND asked a role question in the
            # same message (e.g. "Thursday works. Can you tell me more about the stack?").
            # Check the Info Advisor and, if relevant, include the answer.
            info_rec = self.info_advisor.evaluate(
                self._format_history(), candidate_message, role=self.candidate_position
            )
            position_label = self.candidate_position or ""
            confirmation = (
                f"Perfect! Your {position_label} interview is confirmed for "
                f"{confirmed_slot}. You'll receive a calendar invite shortly."
            )
            if (
                info_rec.get("recommendation") == "retrieve"
                and info_rec.get("content", "N/A") != "N/A"
            ):
                message = f"{confirmation} Also — {info_rec['content']}"
            else:
                message = confirmation
            self.last_confirmed_slot = confirmed_slot
            if self.candidate_position:
                self.scheduling_advisor.mark_slot_booked(confirmed_slot, self.candidate_position)
            return self._record_and_return("end", message, slots=None)

        # Not a confirmation — run normal LLM flow with slot context injected
        return self._llm_turn(
            candidate_message, conversation_time, in_slot_negotiation=True
        )

    def _llm_turn(
        self,
        candidate_message: str,
        conversation_time: Optional[str],
        in_slot_negotiation: bool = False,
    ) -> tuple[str, str]:
        """Full advisor + LLM flow."""
        history_text = self._format_history()

        exit_rec = self.exit_advisor.evaluate(history_text)
        # After "Continue Conversation" the candidate explicitly wants to keep going,
        # so ignore the ExitAdvisor's end recommendation for this one turn.
        if self.in_continuation and exit_rec.get("recommendation") == "end":
            exit_rec = {"recommendation": "continue", "confidence": "low",
                        "reason": "Conversation resumed by candidate"}
        self.in_continuation = False
        sched_rec = self.scheduling_advisor.evaluate(
            history_text,
            conversation_time,
            self.candidate_position,
            last_offered_slots=self.last_offered_slots if in_slot_negotiation else None,
        )
        info_rec = self.info_advisor.evaluate(history_text, candidate_message, role=self.candidate_position)

        position_label = self.candidate_position or "UNKNOWN"
        slot_context = (
            f"CONTEXT: Slots already offered to candidate: {self.last_offered_slots}. "
            "Do NOT repeat the same slots — only offer new ones if different.\n"
            if in_slot_negotiation else ""
        )
        advisor_block = (
            f"Detected Position: {position_label}\n"
            f"{slot_context}"
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
        new_slots = sched_rec.get("slots", "none") if action == "schedule" else None
        return self._record_and_return(action, message, slots=new_slots)

    def _record_and_return(
        self, action: str, message: str, slots: Optional[str] = None
    ) -> tuple[str, str]:
        """Append bot message to history, update state, return (message, action)."""
        self.conversation_history.append({"role": "assistant", "content": message})
        self.last_action = action
        if action == "schedule" and slots:
            self.last_offered_slots = slots
        elif action == "end":
            self.last_offered_slots = None
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
        self.candidate_position = "Python Dev"
        self.last_action = "continue"
        self.last_offered_slots = None
        self.last_confirmed_slot = None
        self.in_continuation = False

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

    def confirm_slot(self, slot: str) -> tuple[str, str]:
        """Directly confirm a slot chosen via UI button — bypasses LLM entirely."""
        position_label = self.candidate_position or ""
        self.conversation_history.append(
            {"role": "user", "content": f"I'll take the {slot} slot."}
        )
        message = (
            f"Your {position_label} interview is confirmed for {slot}. "
            "We look forward to meeting you! You'll receive a calendar invite shortly."
        )
        self.last_confirmed_slot = slot
        if self.candidate_position:
            self.scheduling_advisor.mark_slot_booked(slot, self.candidate_position)
        return self._record_and_return("end", message, slots=None)

    def _is_slot_confirmed(self, message: str) -> str | None:
        """
        Check if the candidate confirmed one of the offered slots.
        Handles explicit picks AND vague acceptance ("ok", "whatever you want", "any works").
        Returns the confirmed slot string, or None if they want different slots.
        """
        first_slot = (
            self.last_offered_slots.split(",")[0].strip()
            if self.last_offered_slots else ""
        )
        response = self.llm.invoke([HumanMessage(content=(
            f"We offered these interview slots: {self.last_offered_slots}\n"
            f'The candidate replied: "{message}"\n'
            "Rules:\n"
            "- If they named or clearly agreed to a specific slot → return that slot datetime.\n"
            f"- If they said 'ok', 'sure', 'whatever you want', 'any works', 'fine', or any "
            f"  vague acceptance → return the FIRST slot: '{first_slot}'.\n"
            "- If they want DIFFERENT slots, more options, or a different time → reply: NO\n"
            "Reply with ONLY the slot datetime (e.g. '2024-01-02 09:00') or NO."
        ))])
        result = response.content.strip()
        return None if result.upper().startswith("NO") else result

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

        position = _keyword_position(candidate_text)
        if position:
            return position

        if len(candidate_text.split()) < 5:
            return None

        response = self.llm.invoke(
            [HumanMessage(content=(
                "A job candidate sent these messages during a recruiting conversation.\n"
                "Are they applying for the Python Developer role?\n"
                "Answer with EXACTLY one of: Python Dev, UNKNOWN\n"
                "No explanation — just the role name.\n\n"
                f"Candidate messages:\n{candidate_text}"
            ))],
        )
        result = response.content.strip()
        return result if result == "Python Dev" else None

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
