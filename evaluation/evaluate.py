
import json
from pathlib import Path

import requests

EVAL_DIR = Path(__file__).parent
DATASET_PATH = EVAL_DIR / "eval_dataset.json"
PDF_PATH = EVAL_DIR / "company_policy.pdf"

API_URL = "http://localhost:8000"

with open(DATASET_PATH, encoding="utf-8") as f:
    dataset = json.load(f)

# Upload the evaluation PDF
with open(PDF_PATH, "rb") as f:
    response = requests.post(
        f"{API_URL}/upload",
        files={"file": ("company_policy.pdf", f, "application/pdf")},
        timeout=120,
    )

response.raise_for_status()
document_id = response.json()["document_id"]

print(f"Evaluation document uploaded: {document_id}\n")

correct_answers = 0
correct_retrievals = 0
retrieval_cases = 0

for case in dataset:
    response = requests.post(
        f"{API_URL}/ask",
        json={
            "question": case["question"],
            "document_id": document_id,
        },
        timeout=180,
    )
    response.raise_for_status()

    result = response.json()
    answer = result["answer"]
    sources = result.get("sources", [])

    expected_answer = case["expected_answer"]
    expected_page = case["expected_page"]

    # Lightweight keyword-based answer check.
    # This is intentionally a simple baseline.
    if case["id"] == 1:
        answer_correct = "three" in answer.lower() or "3 days" in answer.lower()
    elif case["id"] == 2:
        answer_correct = "twenty" in answer.lower() or "20 days" in answer.lower()
    else:
        answer_lower = answer.lower()
        answer_correct = any(
            phrase in answer_lower
            for phrase in [
                "not mentioned",
                "not provided",
                "couldn't find",
                "cannot find",
                "not in the document",
                "does not specify",
                "doesn't specify",
            ]
        )

    correct_answers += int(answer_correct)

    # Check whether the expected page was retrieved.
    # Unanswerable questions are excluded from this metric.
    retrieval_correct = None

    if expected_page is not None:
        retrieval_cases += 1
        retrieval_correct = any(
            source.get("page_number") == expected_page
            for source in sources
        )
        correct_retrievals += int(retrieval_correct)

    print(f"Question: {case['question']}")
    print(f"Answer: {answer}")
    print(f"Answer check: {'PASS' if answer_correct else 'FAIL'}")
    print(
        "Retrieval check:",
        "N/A" if retrieval_correct is None else
        ("PASS" if retrieval_correct else "FAIL")
    )
    print("-" * 60)

total = len(dataset)

print("\nEVALUATION RESULTS")
print(f"Answer checks: {correct_answers}/{total}")
print(f"Answer check rate: {correct_answers / total:.1%}")
print(f"Expected-page retrieval: {correct_retrievals}/{retrieval_cases}")
print(f"Retrieval hit rate: {correct_retrievals / retrieval_cases:.1%}")
