# Exit Advisor — Fine-Tuning Results

## Model

| | |
|---|---|
| **Base model** | `gpt-4.1-2025-04-14` |
| **Fine-tuned model ID** | `ft:gpt-4.1-2025-04-14:chuckybuilder::Dj32h72n` |
| **Method** | Supervised Fine-Tuning (SFT) |
| **Trained tokens** | 17,331 |
| **Job ID** | `ftjob-d4nWIxjY5pxlHefpua5PGyEK` |

---

## Training Data

| | |
|---|---|
| **Source** | `sms_conversations.json` — 15 labeled SMS recruiting conversations |
| **Total examples** | 59 (one per labeled recruiter turn) |
| **Train split (80%)** | 47 examples → `fine_tuning_data/exit_advisor_train.jsonl` |
| **Test split (20%)** | 12 examples → `fine_tuning_data/exit_advisor_test.jsonl` |
| **Labels** | `end` / `continue` (turns labeled `schedule` map to `continue`) |

---

## Training Configuration

```python
client.fine_tuning.jobs.create(
    model="gpt-4.1-2025-04-14",
    training_file=uploaded_file.id,
    method={"type": "supervised"},
    hyperparameters={"n_epochs": 3},
)
```

---

## Evaluation Results (held-out test set — never seen during training)

**Accuracy: 100.0% (12 / 12)**

| # | True Label | Predicted | Correct |
|---|-----------|-----------|---------|
| 01 | continue | continue | ✅ |
| 02 | continue | continue | ✅ |
| 03 | continue | continue | ✅ |
| 04 | end       | end       | ✅ |
| 05 | end       | end       | ✅ |
| 06 | continue | continue | ✅ |
| 07 | continue | continue | ✅ |
| 08 | continue | continue | ✅ |
| 09 | continue | continue | ✅ |
| 10 | continue | continue | ✅ |
| 11 | continue | continue | ✅ |
| 12 | continue | continue | ✅ |

---

## Confusion Matrix

|  | Predicted: **continue** | Predicted: **end** |
|--|:-:|:-:|
| **True: continue** | 10 | 0 |
| **True: end** | 0 | 2 |

No false positives. No false negatives.

---

## How the Fine-Tuned Model Is Used

The `ExitAdvisor` class reads `EXIT_ADVISOR_MODEL` from `.env` at startup.
When set, it uses the fine-tuned model; otherwise it falls back to `gpt-4o-mini`.

```python
# app/modules/exit_advisor/exit_advisor.py
self.model = model or os.getenv("EXIT_ADVISOR_MODEL", "gpt-4o-mini")
```

The fine-tuned model outputs a single word (`end` or `continue`).
The parser handles both this format and the structured base-model format automatically.
