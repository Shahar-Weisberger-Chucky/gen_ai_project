import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ── Prompting strategy: Role + Instructions + Few-Shot + API param (temp=0) ──

SYSTEM_PROMPT = """You are the Conversation Exit Advisor for a Python Developer recruiting chatbot.

ROLE:
Your only job is to decide whether the recruiting conversation should END right now.
You don't write messages — just evaluate and advise.

INSTRUCTIONS:
Look through the conversation for these signals:

End signals (recommend "end"):
  • Candidate explicitly says they're not interested, already found a job, or asks to be removed
  • Interview was confirmed and the recruiter sent a closing message
  • Candidate has been persistently rude or asked to stop contact

Continue signals (recommend "continue"):
  • Candidate is still engaged, asking questions, or open to scheduling
  • Scheduling is in progress but not yet confirmed
  • Conversation just started

When EXIT_ADVISOR_MODEL is set in .env, this advisor uses a fine-tuned model
trained on labeled SMS conversations for better accuracy.

RESPONSE FORMAT (use exactly this structure, nothing else):
RECOMMENDATION: <end|continue>
CONFIDENCE: <high|medium|low>
REASON: <one sentence explaining your decision>
"""

FEW_SHOT = """
--- Example 1 ---
Candidate: Please remove me from your list. Thanks.
RECOMMENDATION: end
CONFIDENCE: high
REASON: Candidate explicitly asked to be removed from the contact list.

--- Example 2 ---
Candidate: I'm no longer interested in the position.
RECOMMENDATION: end
CONFIDENCE: high
REASON: Candidate clearly expressed disinterest.

--- Example 3 ---
Candidate: Tuesday at 10 AM works.
Recruiter: Great, your interview is confirmed. You'll receive a calendar invite shortly.
RECOMMENDATION: end
CONFIDENCE: high
REASON: Interview was confirmed and recruiter sent a closing message.

--- Example 4 ---
Candidate: I have three years of Python experience, mostly with Django.
RECOMMENDATION: continue
CONFIDENCE: high
REASON: Candidate is engaged and sharing relevant background — conversation is ongoing.

--- Example 5 ---
Candidate: What technologies does the team use?
RECOMMENDATION: continue
CONFIDENCE: high
REASON: Candidate is asking a question about the role, showing active interest.
"""


class ExitAdvisor:
    def __init__(self, model: str = None):
        # swap in a fine-tuned model by setting EXIT_ADVISOR_MODEL in .env
        self.model = model or os.getenv("EXIT_ADVISOR_MODEL", "gpt-4o-mini")
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def evaluate(self, conversation_history: str) -> dict:
        prompt = (
            f"{FEW_SHOT}\n\n"
            f"--- Current Conversation ---\n{conversation_history}\n\n"
            "Should the conversation end now?"
        )
        response = self.llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        return self._parse(response.content)

    @staticmethod
    def _parse(text: str) -> dict:
        result = {"recommendation": "continue", "confidence": "low", "reason": ""}

        # fine-tuned model outputs a single word; handle that first
        stripped = text.strip().lower()
        if stripped in ("end", "continue"):
            result["recommendation"] = stripped
            result["confidence"] = "high"
            return result

        # base model outputs RECOMMENDATION / CONFIDENCE / REASON lines
        for line in text.strip().splitlines():
            key, _, val = line.partition(":")
            key = key.strip().upper()
            val = val.strip()
            if key == "RECOMMENDATION" and val.lower() in ("end", "continue"):
                result["recommendation"] = val.lower()
            elif key == "CONFIDENCE":
                result["confidence"] = val.lower()
            elif key == "REASON":
                result["reason"] = val
        return result
