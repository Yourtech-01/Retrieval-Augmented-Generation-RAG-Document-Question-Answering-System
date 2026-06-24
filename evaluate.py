"""
eval/evaluate.py
Evaluates RAG quality using RAGAS metrics.

Metrics measured:
  - faithfulness:       does the answer stick to the context?
  - answer_relevancy:   is the answer actually relevant to the question?
  - context_recall:     did we retrieve the right chunks?
  - context_precision:  are the retrieved chunks actually useful?

Usage:
    python -m eval.evaluate

Outputs:
    eval/results/ragas_scores.json
    eval/results/ragas_report.html
"""

import json
import pathlib
import sys

# Eval test set — replace with your own domain questions
TEST_QUESTIONS = [
    "What is the main purpose described in the document?",
    "What are the key conclusions or findings?",
    "What methodology was used?",
    "What are the limitations mentioned?",
    "What recommendations are made?",
]

# Ground truth answers for your test docs
# Fill these in after reading your documents
GROUND_TRUTHS = [
    ["The main purpose is to..."],       # fill in
    ["The key findings include..."],     # fill in
    ["The methodology involved..."],     # fill in
    ["The limitations include..."],      # fill in
    ["The recommendations are..."],      # fill in
]

OUTPUT_DIR = pathlib.Path("eval/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_evaluation():
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
    except ImportError:
        print("Install RAGAS: pip install ragas datasets")
        sys.exit(1)

    import requests

    print("[eval] Running RAGAS evaluation...")
    print(f"[eval] Evaluating {len(TEST_QUESTIONS)} questions")

    answers   = []
    contexts  = []

    for q in TEST_QUESTIONS:
        try:
            resp = requests.post(
                "http://localhost:8000/ask",
                json={"question": q, "top_k": 5},
                timeout=30,
            )
            data = resp.json()
            answers.append(data["answer"])
            contexts.append([s["excerpt"] for s in data["sources"]])
        except Exception as e:
            print(f"  Error for question '{q[:50]}...': {e}")
            answers.append("")
            contexts.append([""])

    dataset = Dataset.from_dict({
        "question":    TEST_QUESTIONS,
        "answer":      answers,
        "contexts":    contexts,
        "ground_truth": [gt[0] for gt in GROUND_TRUTHS],
    })

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    print("\n=== RAGAS Scores ===")
    result = {}
    for metric, score in scores.items():
        result[metric] = round(float(score), 4)
        print(f"  {metric:25s}: {score:.4f}")

    # Save JSON
    out_json = OUTPUT_DIR / "ragas_scores.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[eval] Scores saved to {out_json}")

    # Save detailed report
    scores.to_pandas().to_html(OUTPUT_DIR / "ragas_report.html", index=False)
    print(f"[eval] HTML report saved to {OUTPUT_DIR / 'ragas_report.html'}")

    return result


if __name__ == "__main__":
    run_evaluation()
