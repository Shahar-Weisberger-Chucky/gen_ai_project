"""
Fine-tuning pipeline for the Exit Advisor.

Workflow (mirrors the course approach):
  1. prepare_training_data()  — builds JSONL and splits 80/20 into train + test files
  2. upload_and_train()       — uploads the train file, kicks off the SFT job
  3. check_job_status(job_id) — poll until status == 'succeeded'
  4. evaluate(model_id)       — run the fine-tuned model on the test set, print accuracy + confusion matrix

After the job finishes, copy the model ID into .env as EXIT_ADVISOR_MODEL.
"""
import os
import json
import random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CONVERSATIONS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "sms_conversations.json"
)
FINE_TUNE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "fine_tuning_data"
)

# Keep this identical to the live ExitAdvisor system prompt so the fine-tuned
# model is trained on the same framing it will see in production.
FINE_TUNE_SYSTEM_PROMPT = (
    "You are the Conversation Exit Advisor for a recruiting chatbot. "
    "Analyse the conversation and decide: should the conversation end NOW? "
    "Reply with exactly one word: end OR continue"
)


def _build_examples(conversations: list) -> list[dict]:
    """Convert labeled recruiter turns into chat-completions JSONL examples."""
    examples = []
    for conv in conversations:
        turns = conv["turns"]
        for idx, turn in enumerate(turns):
            if turn["speaker"] != "recruiter" or turn["label"] is None:
                continue

            history = []
            for t in turns[: idx + 1]:
                role = "assistant" if t["speaker"] == "recruiter" else "user"
                history.append({"role": role, "content": t["text"]})

            # exit advisor only cares about end vs continue
            label = "end" if turn["label"] == "end" else "continue"

            examples.append({
                "messages": [
                    {"role": "system", "content": FINE_TUNE_SYSTEM_PROMPT},
                    *history,
                    {"role": "assistant", "content": label},
                ],
                "_true_label": label,  # kept for eval; stripped before writing JSONL
            })
    return examples


def prepare_training_data(train_path: str = None, test_path: str = None, seed: int = 42) -> tuple[str, str]:
    """
    Read sms_conversations.json, build examples, and split 80/20 into train + test JSONL.
    Returns (train_path, test_path).
    """
    os.makedirs(FINE_TUNE_DIR, exist_ok=True)
    if train_path is None:
        train_path = os.path.join(FINE_TUNE_DIR, "exit_advisor_train.jsonl")
    if test_path is None:
        test_path = os.path.join(FINE_TUNE_DIR, "exit_advisor_test.jsonl")

    with open(CONVERSATIONS_PATH, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    examples = _build_examples(conversations)

    random.seed(seed)
    random.shuffle(examples)

    split = int(len(examples) * 0.8)
    train_examples = examples[:split]
    test_examples = examples[split:]

    def _write(path, exs):
        with open(path, "w", encoding="utf-8") as f:
            for ex in exs:
                # strip the internal _true_label key before writing
                record = {k: v for k, v in ex.items() if k != "_true_label"}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    _write(train_path, train_examples)
    _write(test_path, test_examples)

    print(f"Total examples : {len(examples)}")
    print(f"Train          : {len(train_examples)}  -> {train_path}")
    print(f"Test           : {len(test_examples)}   -> {test_path}")
    return train_path, test_path


def upload_and_train(train_path: str = None) -> str:
    """
    Upload the training JSONL to OpenAI and start a supervised fine-tuning job.
    Returns the job ID. Monitor at https://platform.openai.com/finetune
    """
    if train_path is None:
        train_path, _ = prepare_training_data()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"Uploading {train_path} …")
    with open(train_path, "rb") as f:
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
        f"  EXIT_ADVISOR_MODEL=<model-id-shown-when-done>"
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


def evaluate(fine_tuned_model: str, test_path: str = None) -> dict:
    """
    Run the fine-tuned model on the test set and report accuracy + confusion matrix.

    Mirrors the professor's eval pattern:
      1. Load test JSONL
      2. For each example, strip the last assistant message (the true label) and ask the model
      3. Compare predictions to ground truth
      4. Print accuracy, confusion matrix
    """
    if test_path is None:
        test_path = os.path.join(FINE_TUNE_DIR, "exit_advisor_test.jsonl")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}. Run prepare_training_data() first.")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # load test examples
    test_examples = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            test_examples.append(json.loads(line))

    y_true = []
    y_pred = []

    print(f"Evaluating {fine_tuned_model} on {len(test_examples)} test examples…")
    for i, ex in enumerate(test_examples):
        messages = ex["messages"]
        true_label = messages[-1]["content"].strip().lower()  # last assistant msg = ground truth
        prompt_messages = messages[:-1]                        # everything except the answer

        response = client.chat.completions.create(
            model=fine_tuned_model,
            messages=prompt_messages,
            temperature=0,
            max_tokens=5,
        )
        pred = response.choices[0].message.content.strip().lower()

        y_true.append(true_label)
        y_pred.append(pred)
        print(f"  [{i+1:02d}/{len(test_examples)}] true={true_label:8s}  pred={pred}")

    # accuracy
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / len(y_true)
    print(f"\nAccuracy: {accuracy:.1%}  ({correct}/{len(y_true)})")

    # confusion matrix
    labels = sorted(set(y_true) | set(y_pred))
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    header = f"{'':12s}" + "".join(f"{l:12s}" for l in labels)
    print(header)
    for true_l in labels:
        row = f"{true_l:12s}"
        for pred_l in labels:
            count = sum(t == true_l and p == pred_l for t, p in zip(y_true, y_pred))
            row += f"{count:<12d}"
        print(row)

    return {"accuracy": accuracy, "y_true": y_true, "y_pred": y_pred}


if __name__ == "__main__":
    # ── Step 1: check the completed job and get the model ID ──────────────────
    JOB_ID = "ftjob-d4nWIxjY5pxlHefpua5PGyEK"
    status = check_job_status(JOB_ID)

    model_id = status.get("fine_tuned_model")
    if not model_id:
        print("Job not finished yet — try again in a few minutes.")
    else:
        print(f"\nFine-tuned model: {model_id}")
        print("\nAdd this to your .env file:")
        print(f"  EXIT_ADVISOR_MODEL={model_id}")

        # ── Step 2: evaluate on the test set ─────────────────────────────────
        print("\n--- Running evaluation on test set ---")
        evaluate(model_id)
