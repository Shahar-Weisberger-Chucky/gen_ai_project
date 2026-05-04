import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma

load_dotenv()

CHROMA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "chroma_db"
)

# ── Prompting strategy: Role + Instructions + RAG context + API param ────────

SYSTEM_PROMPT = """You are the Conversation Info Advisor for a tech company recruiting chatbot.
The company hires for four roles: ML Engineer, SQL Developer, Data Analyst, and Python Developer.

ROLE:
Check whether the candidate is asking a question about the job that you should answer.
If yes, pull the answer from the retrieved job description context.

INSTRUCTIONS:
  • Return recommendation "retrieve" ONLY if the candidate asked a real question about
    the role, company, tech stack, salary, work model, or anything position-related
  • Return recommendation "none" if:
      - The message is NOT a question about the position
      - The question is about scheduling (that's the Scheduling Advisor's job)
      - The candidate is just introducing themselves or making small talk
  • Keep the answer short (1–2 sentences, SMS style)
  • Stick to the retrieved context — don't make things up

RESPONSE FORMAT (use exactly this structure):
RECOMMENDATION: <retrieve|none>
CONTENT: <your answer, or "N/A">
"""

FEW_SHOT = """
--- Example 1 ---
Candidate: What technologies does the team use?
Context: The stack is Python 3, FastAPI, PostgreSQL, Docker, and AWS ECS.
RECOMMENDATION: retrieve
CONTENT: We use Python 3 with FastAPI for the backend, PostgreSQL, Docker, and deploy to AWS ECS.

--- Example 2 ---
Candidate: I have five years of Python experience.
RECOMMENDATION: none
CONTENT: N/A

--- Example 3 ---
Candidate: Is the position remote or hybrid?
Context: Hybrid work model — at least two days remote per week.
RECOMMENDATION: retrieve
CONTENT: The role is hybrid, with at least two remote days per week.

--- Example 4 ---
Candidate: Can we meet next Tuesday?
RECOMMENDATION: none
CONTENT: N/A
"""


class InfoAdvisor:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        self._vectorstore = None

    def _load_vectorstore(self) -> Chroma | None:
        if self._vectorstore is None:
            chroma_path = os.path.abspath(CHROMA_DIR)
            if os.path.exists(chroma_path):
                try:
                    self._vectorstore = Chroma(
                        persist_directory=chroma_path,
                        embedding_function=self.embeddings,
                    )
                except Exception:
                    pass
        return self._vectorstore

    def _retrieve_context(self, question: str) -> str:
        vs = self._load_vectorstore()
        if vs is None:
            return "(knowledge base not yet built — run embedding.py first)"
        try:
            docs = vs.similarity_search(question, k=3)
            return "\n\n".join(d.page_content for d in docs)
        except Exception:
            return ""

    def evaluate(self, conversation_history: str, latest_message: str) -> dict:
        context = self._retrieve_context(latest_message)

        prompt = (
            f"{FEW_SHOT}\n\n"
            f"--- Current Conversation ---\n{conversation_history}\n\n"
            f"Latest candidate message: {latest_message}\n\n"
            f"Retrieved context from knowledge base:\n{context}\n\n"
            "Does the candidate need an answer about the position?"
        )

        response = self.llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        return self._parse(response.content)

    @staticmethod
    def _parse(text: str) -> dict:
        result = {"recommendation": "none", "content": "N/A"}
        for line in text.strip().splitlines():
            key, _, val = line.partition(":")
            key = key.strip().upper()
            val = val.strip()
            if key == "RECOMMENDATION":
                result["recommendation"] = val.lower()
            elif key == "CONTENT":
                result["content"] = val
        return result
