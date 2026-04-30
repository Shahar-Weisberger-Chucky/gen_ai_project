"""
Fine-tuning pipeline for the Exit Advisor.

Three steps:
1. prepare_training_data() — converts sms_conversations.json into JSONL,
   one example per labeled recruiter turn (end / continue)
2. upload_and_train() — uploads the JSONL and kicks off the fine-tuning job
3. check_job_status(job_id) — check how the job is going

After the job finishes, copy the model ID into .env as EXIT_ADVISOR_MODEL.
"""
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CONVERSATIONS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "sms_conversations.json"
)
FINE_TUNE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "fine_tuning_data"
)

# the system prompt every training example gets — keep it identical to the live advisor
FINE_TUNE_SYSTEM_PROMPT = (
    "You are the Conversation Exit Advisor for a recruiting chatbot. "
    "Analyse the conversation and decide: should the conversation end NOW? "
    "Reply with exactly one word: end OR continue"
)


def prepare_training_data(output_path: str = None) -> str:
    """
    Read sms_conversations.json and write a JSONL file for fine-tuning.
    Only recruiter turns with a label become training examples.
    Returns the path to the output file.
    """
    os.makedirs(FINE_TUNE_DIR, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(FINE_TUNE_DIR, "exit_advisor_train.jsonl")

    with open(CONVERSATIONS_PATH, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    examples = []

    for conv in conversations:
        turns = conv["turns"]

        for idx, turn in enumerate(turns):
            if turn["speaker"] != "recruiter" or turn["label"] is None:
                continue

            # build the full message history up to this turn
            history = []
            for t in turns[: idx + 1]:
                role = "assistant" if t["speaker"] == "recruiter" else "user"
                history.append({"role": role, "content": t["text"]})

            label = "end" if turn["label"] == "end" else "continue"

            examples.append({
                "messages": [
                    {"role": "system", "content": FINE_TUNE_SYSTEM_PROMPT},
                    *history,
                    {"role": "assistant", "content": label},
                ]
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Prepared {len(examples)} training examples → {output_path}")
    return output_path


def upload_and_train(training_file_path: str = None) -> str:
    """
    Upload the JSONL to OpenAI and start a supervised fine-tuning job.
    Returns the job ID. Monitor at https://platform.openai.com/finetune
    """
    if training_file_path is None:
        training_file_path = prepare_training_data()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"Uploading {training_file_path} …")
    with open(training_file_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="fine-tune")
    print(f"File uploaded: {uploaded.id}")

    job = client.fine_tuning.jobs.create(
        training_file=uploaded.id,
        model="gpt-4.1-2025-04-14",
        method={"type": "supervised"},
        hyperparameters={"n_epochs": 3},
    )
    print(f"Fine-tuning job created: {job.id}")
    print("Monitor at: https://platform.openai.com/finetune")
    print(
        "Once complete, add this to .env:\n"
        f"  EXIT_ADVISOR_MODEL={job.fine_tuned_model or '<model-id-assigned-when-done>'}"
    )
    return job.id


def check_job_status(job_id: str) -> dict:
    """Quick status check on a running fine-tuning job."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    job = client.fine_tuning.jobs.retrieve(job_id)
    status = {
        "id": job.id,
        "status": job.status,
        "fine_tuned_model": job.fine_tuned_model,
        "trained_tokens": getattr(job, "trained_tokens", None),
    }
    print(status)
    return status


if __name__ == "__main__":
    prepare_training_data()
    print("\nTraining data ready. Call upload_and_train() to start fine-tuning.")
