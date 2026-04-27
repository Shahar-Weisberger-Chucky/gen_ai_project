from dotenv import load_dotenv
import os
import urllib

import pandas as pd
from sqlalchemy import create_engine
from langchain.tools import tool
from langchain.agents import create_agent


load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    raise ValueError("Missing OPENAI_API_KEY in .env file")


def get_engine():
    """
    Creates a SQLAlchemy connection engine to the local SQL Server Tech database.
    """
    params = urllib.parse.quote_plus(
        "Driver={ODBC Driver 18 for SQL Server};"
        "Server=localhost,1433;"
        "Database=Tech;"
        "UID=sa;"
        "PWD=SqlPassword123!;"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
    )

    return create_engine(f"mssql+pyodbc:///?odbc_connect={params}")


def get_schedule():
    """
    Loads all interview schedule rows from dbo.Schedule into a pandas DataFrame.
    """
    engine = get_engine()
    query = "SELECT * FROM dbo.Schedule"
    return pd.read_sql(query, engine)


def normalize_position(user_position: str):
    """
    Converts free-text position input into one of the supported database position names.
    """
    pos = user_position.lower()

    if "python" in pos:
        return "Python Dev"

    if "sql" in pos:
        return "Sql Dev"

    if "analyst" in pos or "data" in pos:
        return "Analyst"

    if "machine" in pos or "ml" in pos:
        return "ML"

    return None


@tool
def suggest_slots(user_position: str) -> str:
    """
    Finds the first 3 available interview slots for a requested job position.
    Input examples: Python, SQL, Analyst, Data, ML, Machine Learning.
    """
    df = get_schedule()

    normalized_position = normalize_position(user_position)

    if normalized_position is None:
        return f"Could not understand the position: {user_position}"

    filtered_df = df[
        (df["position"] == normalized_position)
        & (df["available"] == True)
    ].copy()

    sorted_df = filtered_df.sort_values(by=["date", "time"])
    top_3 = sorted_df.head(3)

    if top_3.empty:
        return f"No available slots found for {normalized_position}."

    top_3["date"] = pd.to_datetime(top_3["date"]).dt.strftime("%Y-%m-%d")
    top_3["time"] = top_3["time"].astype(str)

    return top_3[["date", "time", "position"]].to_json(orient="records")


SYSTEM_PROMPT = """
You are an SMS-based chatbot for job candidates.

Context:
The company is scheduling interviews for technical roles.

Your job:
- Help candidates schedule interview slots.
- If the candidate asks about availability, interview times, appointments, slots, or roles,
  use the suggest_slots tool.
- If the candidate mentions Python, SQL, Analyst, Data, ML, or Machine Learning,
  infer the relevant position.
- Keep answers short, friendly, and suitable for SMS.
- When showing slots, format them clearly as Option 1, Option 2, Option 3.
"""


agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[suggest_slots],
    system_prompt=SYSTEM_PROMPT,
)


session_store = {}


def get_messages(session_id: str):
    """
    Returns the conversation history for a specific session.
    Creates an empty history if this is a new session.
    """
    if session_id not in session_store:
        session_store[session_id] = []

    return session_store[session_id]


def orchestrate_conversation(user_input: str, session_id: str = "user1") -> str:
    """
    Main function called by Streamlit.
    Sends the user message to the LangChain agent and returns the assistant answer.
    """
    messages = get_messages(session_id)

    messages.append({
        "role": "user",
        "content": user_input
    })

    result = agent.invoke({
        "messages": messages
    })

    assistant_message = result["messages"][-1]
    assistant_text = assistant_message.content

    messages.append({
        "role": "assistant",
        "content": assistant_text
    })

    return assistant_text