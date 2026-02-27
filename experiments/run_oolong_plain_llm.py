"""
OOLONG Experiment with Plain LLM (No RLM)

Baseline: directly feed the full context + question to the LLM in a single call,
without any RLM recursive decomposition or REPL environment.

Uses the same data loading and evaluation as run_oolong_experiment.py
so results are directly comparable.
"""

import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from run_oolong_experiment import evaluate_oolong_response


def run_oolong_plain_experiment(
    num_samples: int = 50,
    min_context_len: int = 1024,
    max_context_len: int = 65536,
    output_name: str = "oolong_plain_llm",
    dataset_filter: str = "trec_coarse",
):
    """
    Run OOLONG experiment with a plain LLM call (no RLM).
    Same data, same evaluation — only the inference method differs.
    """
    from datasets import load_dataset

    print("=" * 60)
    print("OOLONG Experiment — Plain LLM (no RLM)")
    print(f"Dataset filter: {dataset_filter}")
    print(f"samples={num_samples}")
    print("=" * 60)

    # --- data ---
    print("Loading OOLONG-synth dataset from HuggingFace...")
    try:
        dataset = load_dataset("oolongbench/oolong-synth", split="test")
        print(f"Loaded {len(dataset)} total samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please make sure you have internet access and datasets installed.")
        return None

    def filter_by_dataset(x):
        return x.get("dataset", "").lower() == dataset_filter.lower()

    try:
        filtered = dataset.filter(filter_by_dataset)
        print(f"Filtered to {len(filtered)} samples from '{dataset_filter}' split")
    except Exception as e:
        print(f"Warning: Could not filter by dataset: {e}")
        filtered = dataset

    def filter_by_length(x):
        ctx_len = x.get("context_len", len(x.get("context_window_text", "")))
        return min_context_len <= ctx_len <= max_context_len

    try:
        filtered = filtered.filter(filter_by_length)
        print(f"After context length filter: {len(filtered)} samples")
    except:
        pass

    samples = list(filtered)[:num_samples]
    print(f"Using {len(samples)} samples for experiment")

    if len(samples) == 0:
        print("ERROR: No samples found! Check the dataset_filter value.")
        return None

    # --- LLM client ---
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    model_name = os.getenv("MODEL_NAME", "deepseek-chat")

    if not api_key:
        raise ValueError("No API key found! Set DEEPSEEK_API_KEY or OPENAI_API_KEY.")

    client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"Model: {model_name}")

    # --- run ---
    results: list[dict] = []
    total_score = 0.0
    correct_count = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i+1}/{len(samples)} ---")

        context = sample.get("context_window_text", sample.get("context", ""))
        question = sample.get("question", "")
        gold_answer = sample.get("answer", "")
        answer_type = sample.get("answer_type", None)
        dataset_name = sample.get("dataset", "unknown")

        print(f"Dataset: {dataset_name}")
        print(f"Question: {question[:100]}...")
        print(f"Gold: {gold_answer[:50] if isinstance(gold_answer, str) else gold_answer}...")

        try:
            t0 = time.time()
            response_obj = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": f"{context}\n\n{question}",
                    }
                ],
                temperature=0,
                max_tokens=512,
            )
            elapsed = time.time() - t0

            response = response_obj.choices[0].message.content or ""
            eval_result = evaluate_oolong_response(response, gold_answer, answer_type)

            total_score += eval_result["score"]
            if eval_result["correct"]:
                correct_count += 1
                print(f"✓ CORRECT: {eval_result['extracted_answer']}")
            else:
                print(f"✗ Score: {eval_result['score']:.2f}")
                print(f"  Extracted: {eval_result['extracted_answer'][:100]}")
                print(f"  Response: {response[:150]}...")

            usage_info = {}
            if response_obj.usage:
                inp = response_obj.usage.prompt_tokens or 0
                out = response_obj.usage.completion_tokens or 0
                total_input_tokens += inp
                total_output_tokens += out
                usage_info = {
                    "input_tokens": inp,
                    "output_tokens": out,
                    "total_tokens": inp + out,
                }

            results.append({
                "sample_id": i,
                "dataset": dataset_name,
                "question": question,
                "gold_answer": str(gold_answer),
                "llm_response": response,
                "score": eval_result["score"],
                "correct": eval_result["correct"],
                "extracted_answer": eval_result["extracted_answer"],
                "parse_confidence": eval_result["parse_confidence"],
                "execution_time": elapsed,
                "usage": usage_info,
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "sample_id": i,
                "dataset": dataset_name,
                "question": question,
                "gold_answer": str(gold_answer),
                "llm_response": f"ERROR: {str(e)}",
                "score": 0.0,
                "correct": False,
                "extracted_answer": "",
                "parse_confidence": "error",
                "execution_time": 0,
            })

    # --- metrics ---
    accuracy = correct_count / len(samples) * 100
    avg_score = total_score / len(samples) * 100
    avg_time = sum(r["execution_time"] for r in results) / len(results)
    total_tokens = total_input_tokens + total_output_tokens

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Exact Match Accuracy: {accuracy:.1f}% ({correct_count}/{len(samples)})")
    print(f"Average Score: {avg_score:.1f}%")
    print(f"Avg Execution Time: {avg_time:.2f}s")
    if total_tokens > 0:
        print(f"Total tokens: in={total_input_tokens}, out={total_output_tokens}, "
              f"~{total_tokens} total")

    # --- save ---
    output = {
        "experiment": output_name,
        "model": model_name,
        "method": "plain_llm",
        "dataset_filter": dataset_filter,
        "num_samples": len(samples),
        "accuracy": accuracy,
        "avg_score": avg_score,
        "correct_count": correct_count,
        "avg_execution_time": avg_time,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "results": results,
    }

    output_path = Path(f"results/{output_name}_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OOLONG — Plain LLM baseline")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--dataset", type=str, default="trec_coarse")
    parser.add_argument("--name", type=str, default="oolong_plain_llm")
    args = parser.parse_args()

    run_oolong_plain_experiment(
        num_samples=args.samples,
        output_name=args.name,
        dataset_filter=args.dataset,
    )
