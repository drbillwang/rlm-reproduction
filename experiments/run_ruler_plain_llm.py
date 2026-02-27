"""
RULER S-NIAH Experiment with Plain LLM (No RLM)

Baseline: directly feed the full context + query to the LLM in a single call,
without any RLM recursive decomposition or REPL environment.

Uses the same data loading and evaluation as run_ruler_experiment.py
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
from run_ruler_experiment import (
    generate_ruler_data_official,
    generate_ruler_data_manual,
    load_ruler_data,
    evaluate_ruler_response,
)


def run_ruler_plain_experiment(
    num_samples: int = 50,
    max_seq_length: int = 4096,
    output_name: str = "ruler_plain_llm",
):
    """
    Run RULER experiment with a plain LLM call (no RLM).
    Same data, same evaluation — only the inference method differs.
    """

    print("=" * 60)
    print("RULER S-NIAH Experiment — Plain LLM (no RLM)")
    print(f"samples={num_samples}, length={max_seq_length}")
    print("=" * 60)

    # --- data (reuse existing helpers) ---
    data_dir = Path("data/ruler")
    try:
        data_file = generate_ruler_data_official(
            output_dir=str(data_dir),
            max_seq_length=max_seq_length,
            num_samples=num_samples,
            tokenizer_type="openai",
            tokenizer_path="cl100k_base",
        )
    except Exception as e:
        print(f"Official generation failed: {e}")
        print("Using manual generation...")
        data_file = generate_ruler_data_manual(
            output_dir=str(data_dir),
            max_seq_length=max_seq_length,
            num_samples=num_samples,
        )

    samples = load_ruler_data(data_file)[:num_samples]
    print(f"Loaded {len(samples)} samples")

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
    total_cost = 0.0

    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i+1}/{len(samples)} ---")

        input_text = sample.get("input", "")
        outputs = sample.get("outputs", sample.get("answer", [""]))
        if isinstance(outputs, str):
            outputs = [outputs]

        key = sample.get("needle_key", "unknown")
        value = sample.get("needle_value", outputs[0] if outputs else "")
        print(f"Key: {key}, Expected: {value}")

        query = (
            f"What are all the special magic numbers for {key} "
            "mentioned in the provided text?"
        )

        try:
            t0 = time.time()
            response_obj = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": f"{input_text}\n\n{query}",
                    }
                ],
                temperature=0,
                max_tokens=512,
            )
            elapsed = time.time() - t0

            response = response_obj.choices[0].message.content or ""
            eval_result = evaluate_ruler_response(response, outputs)

            total_score += eval_result["score"]
            if eval_result["correct"]:
                correct_count += 1
                print(f"✓ CORRECT: {eval_result['found']}")
            else:
                print(f"✗ PARTIAL: found {eval_result['found']} / expected {outputs}")
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
                "needle_key": key,
                "gold_answers": outputs,
                "llm_response": response,
                "score": eval_result["score"],
                "correct": eval_result["correct"],
                "found": eval_result["found"],
                "execution_time": elapsed,
                "usage": usage_info,
            })

        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()
            results.append({
                "sample_id": i,
                "needle_key": key,
                "gold_answers": outputs,
                "llm_response": f"ERROR: {str(e)}",
                "score": 0.0,
                "correct": False,
                "found": [],
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
        "max_seq_length": max_seq_length,
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

    parser = argparse.ArgumentParser(description="RULER S-NIAH — Plain LLM baseline")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--name", type=str, default="ruler_plain_llm")
    args = parser.parse_args()

    run_ruler_plain_experiment(
        num_samples=args.samples,
        max_seq_length=args.length,
        output_name=args.name,
    )
