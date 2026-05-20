import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma

load_dotenv()

CHROMA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "chroma_db"
)

# Must stay in sync with COLLECTION_NAME in embedding.py.
COLLECTION_NAME = "job_descriptions"

# Cosine distance threshold (0 = identical, 2 = opposite).
# Chunks scoring above this are considered irrelevant and are dropped.
# This prevents the LLM from hallucinating answers based on off-topic context.
SCORE_THRESHOLD = 1.0

# ── Prompting strategy: Role + Instructions + RAG context + API param ────────

SYSTEM_PROMPT = """You are the Conversation Info Advisor for a tech company recruiting chatbot.

ROLE:
Check whether the candidate is asking a question about the job that you should answer.
If yes, pull the answer from the retrieved job description context.

IMPORTANT: The knowledge base currently contains information for the Python Developer
role only. Only answer questions if the retrieved context is relevant.

INSTRUCTIONS:
  • Return recommendation "retrieve" ONLY if the candidate asked a real question about
    the role, company, tech stack, salary, work model, or anything position-related
  • Return recommendation "none" if:
      - The message is NOT a question about the position
      - The question is about scheduling (that's the Scheduling Advisor's job)
      - The candidate is just introducing themselves or making small talk
      - No relevant context was retrieved
  • Keep the answer short (1–2 sentences, SMS style)
  • Stick strictly to the retrieved context — do NOT make things up

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

--- Example 5 ---
Candidate: What is the salary range?
Context: (no relevant context retrieved)
RECOMMENDATION: none
CONTENT: N/A
"""


class InfoAdvisor:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self._vectorstore = None

    def _load_vectorstore(self) -> Chroma | None:
        if self._vectorstore is None:
            chroma_path = os.path.abspath(CHROMA_DIR)
            if not os.path.exists(chroma_path):
                print("[InfoAdvisor] Warning: chroma_db/ not found — run embedding.py first.")
                return None
            try:
                self._vectorstore = Chroma(
                    persist_directory=chroma_path,
                    embedding_function=self.embeddings,
                    collection_name=COLLECTION_NAME,
                )
            except Exception as e:
                print(f"[InfoAdvisor] Error loading vectorstore: {e}")
        return self._vectorstore

    def _retrieve_context(self, question: str, role: str | None = None) -> str:
        """
        Retrieve relevant chunks for the question.

        Args:
            question: The candidate's message to search against.
            role:     Optional position filter (e.g. "Python Dev"). When provided,
                      only chunks tagged with that role are searched. Leave None
                      to search across all roles (useful when position is unknown).
        """
        vs = self._load_vectorstore()
        if vs is None:
            return ""

        try:
            search_kwargs = {"k": 3}
            if role:
                search_kwargs["filter"] = {"role": role}

            results = vs.similarity_search_with_score(question, **search_kwargs)

            # Drop chunks that are too far from the query (likely irrelevant).
            relevant = [
                doc.page_content
                for doc, score in results
                if score <= SCORE_THRESHOLD
            ]
            return "\n\n".join(relevant) if relevant else ""
        except Exception as e:
            print(f"[InfoAdvisor] Retrieval error: {e}")
            return ""

    def evaluate(
        self,
        conversation_history: str,
        latest_message: str,
        role: str | None = None,
    ) -> dict:
        """
        Args:
            conversation_history: Full formatted conversation so far.
            latest_message:       The candidate's most recent message.
            role:                 Detected position (e.g. "Python Dev"). Used to
                                  filter retrieved chunks to the correct role.
        """
        context = self._retrieve_context(latest_message, role=role)
        context_text = context if context else "(no relevant context retrieved)"

        prompt = (
            f"{FEW_SHOT}\n\n"
            f"--- Current Conversation ---\n{conversation_history}\n\n"
            f"Latest candidate message: {latest_message}\n\n"
            f"Retrieved context from knowledge base:\n{context_text}\n\n"
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
