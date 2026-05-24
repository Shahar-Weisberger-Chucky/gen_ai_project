<!-- PROJECT LOGO -->
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Logo" width="120" height="120">
</p>

<h1 align="center">Python Developer Recruiter Bot</h1>

<p align="center">
  A multi-agent SMS-style chatbot that recruits Python Developer candidates<br>
  <a href="https://github.com/Shahar-Weisberger-Chucky/gen_ai_project">View Repo</a>
  ·
  <a href="https://github.com/Shahar-Weisberger-Chucky/gen_ai_project/issues">Report Bug</a>
  ·
  <a href="https://github.com/Shahar-Weisberger-Chucky/gen_ai_project/issues">Request Feature</a>
</p>

---

<br></br>

## Table of Contents

- [About The Project](#about-the-project)
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Code Examples](#code-examples)
- [Project Structure](#project-structure)
- [To-Do List](#to-do-list)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

<br></br>

## About The Project

> I built this as my final project for the Generative AI & LLMs hands-on course.
> The idea is a recruiting chatbot that talks to Python Developer candidates over SMS —
> it collects their background, answers questions about the job from a PDF knowledge base,
> checks the recruiter's actual calendar for open slots, and knows when to close the conversation.

<div style="background: #272822; color: #f8f8f2; padding: 10px; border-radius: 8px;">
  <b>Technologies:</b> Python, LangChain, OpenAI API, Chroma (vector DB), SQL Server, Streamlit
</div>

### Architecture

The system is split into four agents — one main orchestrator and three specialized advisors:

| Agent | What it does |
|---|---|
| **Main Agent** | Runs the conversation turn by turn. Decides: `continue` / `schedule` / `end` |
| **Exit Advisor** | Reads the conversation and decides if it's time to close — supports a fine-tuned model |
| **Scheduling Advisor** | Calls an SQL Server tool (via LangChain `@tool` + `AgentExecutor`) to find open slots |
| **Info Advisor** | RAG over the job description PDF to answer candidate questions |

---

<br></br>

## Features

- [x] Multi-agent orchestration with LangChain + OpenAI
- [x] RAG pipeline — job description PDF embedded into Chroma vector database
- [x] SQL Server function calling via LangChain `@tool` and `AgentExecutor`
- [x] Fine-tuning pipeline for the Exit Advisor (`gpt-4.1-2025-04-14`)
- [x] Streamlit chat UI (SMS-style PoC)
- [x] Evaluation notebook — Accuracy & Confusion Matrix on labeled conversations
- [x] Role, instruction, few-shot, and API-parameter prompting strategies
- [x] Modern Python project structure with virtual environment
- [ ] Streamlit Community Cloud deployment _(coming soon!)_

---

<br></br>

## Getting Started

### Prerequisites

- Python >= 3.10
- pip
- SQL Server (with SSMS) — run `db_Tech.sql` once to create the `Tech` database
- OpenAI API key

### Installation

```bash
git clone https://github.com/Shahar-Weisberger-Chucky/gen_ai_project.git
cd gen_ai_project
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root with:

```env
OPENAI_API_KEY=sk-...

SQL_SERVER=localhost
SQL_DATABASE=Tech
SQL_DRIVER={ODBC Driver 17 for SQL Server}
SQL_TRUSTED_CONNECTION=yes
```

### One-time Setup

**1. Create the SQL Server database**

Open `db_Tech.sql` in SQL Server Management Studio and execute it.
This creates the `Tech` database with a `Schedule` table seeded with interview availability.

**2. Build the Chroma vector database**

```bash
python -m app.modules.embedding.embedding
```

This embeds `Python Developer Job Description.pdf` into a local `chroma_db/` folder.

---

<br></br>

## Usage

### Run the Streamlit UI

```bash
streamlit run streamlit_app/streamlit_main.py
```

Open your browser at `http://localhost:8501`.

### Run the CLI (for quick testing)

```bash
python -m app.main
```

### Run the Evaluation Notebook

Open `tests/test_evals.ipynb` in Jupyter and run all cells.

### Fine-tune the Exit Advisor

The fine-tuning pipeline is already complete. Results: **100% accuracy on 12 test examples**.
See `tests/fine_tuning_results.md` for full details.

The fine-tuned model is already wired into `.env`:
```env
EXIT_ADVISOR_MODEL=ft:gpt-4.1-2025-04-14:chuckybuilder::Dj32h72n
```

To re-run the fine-tuning pipeline from scratch:
```bash
python -m app.modules.fine_tuning.fine_tuning
```
This generates the JSONL train/test split, uploads to OpenAI, starts the SFT job, and evaluates once complete.

---

<br></br>

## Screenshots

_Add screenshots here after deployment._

---

<br></br>

## Code Examples

### Start a conversation programmatically

```python
from app.main import create_agent

agent = create_agent()
message, action = agent.process_turn("I have 4 years of Python experience.")
print(f"[{action}] {message}")
```

### Query available interview slots directly

```python
from app.modules.scheduling_advisor.scheduling_advisor import get_available_slots

slots = get_available_slots.invoke({"reference_date": "2025-04-10"})
print(slots)
```

### Build the vector database

```python
from app.modules.embedding.embedding import build_vectorstore

vectorstore = build_vectorstore()   # reads PDF, embeds, saves to chroma_db/
```

---

<br></br>

## Project Structure

```text
final_project/
├── .gitignore
├── .env                              # Environment variables (not committed)
├── requirements.txt
├── README.md
├── db_Tech.sql                       # SQL Server schema — run in SSMS
├── sms_conversations.json            # Labeled conversation dataset
├── Python Developer Job Description.pdf
│
├── app/                              # Main application package
│   ├── __init__.py
│   ├── main.py                       # Entry point (CLI)
│   └── modules/
│       ├── __init__.py
│       ├── main_agent/
│       │   ├── __init__.py
│       │   └── main_agent.py         # Orchestrator agent
│       ├── exit_advisor/
│       │   ├── __init__.py
│       │   └── exit_advisor.py       # End-conversation detector (fine-tunable)
│       ├── scheduling_advisor/
│       │   ├── __init__.py
│       │   └── scheduling_advisor.py # SQL tool calling + AgentExecutor
│       ├── info_advisor/
│       │   ├── __init__.py
│       │   └── info_advisor.py       # RAG over job description
│       ├── embedding/
│       │   ├── __init__.py
│       │   └── embedding.py          # PDF → Chroma vector DB
│       └── fine_tuning/
│           ├── __init__.py
│           └── fine_tuning.py        # Exit Advisor fine-tuning pipeline
│
├── streamlit_app/                    # Streamlit UI
│   ├── __init__.py
│   └── streamlit_main.py
│
└── tests/
    └── test_evals.ipynb              # Accuracy + Confusion Matrix evaluation
```

---

<br></br>

## To-Do List

- [x] Project scaffold & Git setup
- [x] SQL Server schema (`db_Tech.sql`)
- [x] Embedding pipeline (PDF → Chroma)
- [x] Exit Advisor (with fine-tuning pipeline)
- [x] Scheduling Advisor (SQL tool calling via AgentExecutor)
- [x] Info Advisor (RAG)
- [x] Main Agent orchestration
- [x] Streamlit chat UI
- [x] Evaluation notebook (Accuracy, Confusion Matrix)
- [x] Fine-tune Exit Advisor on labeled data (100% accuracy on test set — see `tests/fine_tuning_results.md`)
- [ ] Deploy to Streamlit Community Cloud

---

<br></br>

## Contributing

Contributions are **welcome**! Please open an issue or pull request.

---

<br></br>

## License

Distributed under the MIT License.

---

<br></br>

## Contact

**Shahar Weisberger-Chucky**  
Project Link: [https://github.com/Shahar-Weisberger-Chucky/gen_ai_project](https://github.com/Shahar-Weisberger-Chucky/gen_ai_project)

---

<br></br>

## Acknowledgments

- [LangChain](https://python.langchain.com/)
- [OpenAI API](https://platform.openai.com/docs/overview)
- [Chroma](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)

---
