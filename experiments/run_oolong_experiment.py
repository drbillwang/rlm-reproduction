"""
OOLONG Experiment with RLM

OOLONG is a benchmark for evaluating long-context reasoning and aggregation.
Dataset: https://huggingface.co/datasets/oolongbench/oolong-synth

Paper: "We focus specifically on the trec_coarse split, a set of 50 tasks 
over a dataset of questions with semantic labels. Each task requires using 
nearly all entries of the dataset, and therefore scales linearly in 
processing complexity relative to the input length."

This script evaluates RLM on OOLONG with proper accuracy metrics.
"""

import os
import sys
import json
import re
import ast
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def parse_oolong_answer(answer_str: str):
    """Parse OOLONG answer from string format."""
    try:
        if "datetime" in answer_str:
            return datetime.strptime(answer_str, "[datetime.date(%Y, %m, %d)]")
        return ast.literal_eval(answer_str)[0]
    except:
        return answer_str


def evaluate_oolong_response(response: str, gold_answer: str, answer_type: str = None) -> dict:
    """
    Evaluate OOLONG response.
    
    Based on oolong/src/eval/eval_helpers.py
    
    Paper: "We report scoring based on the original paper, which scores 
    numerical answers as score(y_hat) = 0.75^|y - y_hat| and other answers 
    as exact match."
    """
    response = response.strip()
    parse_confidence = "low"
    
    # Try to extract answer from response
    if ":" not in response:
        if len(response) < 20:
            extracted = response
        else:
            extracted = response.split()[-1]
    else:
        extracted = response.split(":")[-1].strip()
        extracted = extracted.replace("*", "").replace("[", "").replace("]", "")
        parse_confidence = "med"
    
    if "User:" in response or "Answer:" in response or "Date:" in response or "Label" in response:
        parse_confidence = "high"
    
    if len(extracted) < 20:
        parse_confidence = "vhigh"
    
    # Handle special answer types
    if "more common" in extracted:
        extracted = "more common"
    elif "less common" in extracted:
        extracted = "less common"
    elif "same frequency" in extracted:
        extracted = "same frequency"
    
    # Calculate score
    score = 0.0
    
    try:
        gold = parse_oolong_answer(gold_answer)
        
        if str(extracted) == str(gold):
            score = 1.0
        elif str(extracted) in ['more common', 'less common', 'same frequency']:
            if str(extracted) in str(gold):
                score = 1.0
        elif answer_type == "ANSWER_TYPE.NUMERIC":
            try:
                extracted_num = int(extracted)
                gold_num = int(gold)
                # Paper formula: score = 0.75^|y - y_hat|
                score = 0.75 ** abs(gold_num - extracted_num)
            except:
                parse_confidence = "low"
        elif answer_type == "ANSWER_TYPE.DATE":
            try:
                import dateutil
                extracted_date = dateutil.parser.parse(extracted)
                if extracted_date == gold:
                    score = 1.0
            except:
                parse_confidence = "low"
    except Exception as e:
        pass
    
    return {
        "score": score,
        "extracted_answer": extracted,
        "parse_confidence": parse_confidence,
        "correct": score == 1.0
    }


def run_oolong_experiment(
    max_depth: int = 1,
    num_samples: int = 50,  # Paper uses 50 tasks
    min_context_len: int = 1024,
    max_context_len: int = 65536,
    output_name: str = "oolong_depth1",
    dataset_filter: str = "trec_coarse"  # Paper focuses on trec_coarse
):
    """
    Run OOLONG experiment with RLM.
    
    Paper specifies using trec_coarse split with 50 tasks.
    """
    from datasets import load_dataset
    from rlm import RLM
    from rlm.logger import RLMLogger
    
    print("=" * 60)
    print(f"OOLONG Experiment with RLM")
    print(f"Dataset filter: {dataset_filter}")
    print(f"max_depth={max_depth}, samples={num_samples}")
    print("=" * 60)
    
    # Load OOLONG-synth dataset
    print("Loading OOLONG-synth dataset from HuggingFace...")
    try:
        dataset = load_dataset("oolongbench/oolong-synth", split="test")
        print(f"Loaded {len(dataset)} total samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please make sure you have internet access and datasets installed:")
        print("  pip install datasets")
        return None
    
    # Filter by dataset (trec_coarse as per paper)
    def filter_by_dataset(x):
        return x.get("dataset", "").lower() == dataset_filter.lower()
    
    try:
        filtered = dataset.filter(filter_by_dataset)
        print(f"Filtered to {len(filtered)} samples from '{dataset_filter}' split")
    except Exception as e:
        print(f"Warning: Could not filter by dataset: {e}")
        print("Using all samples...")
        filtered = dataset
    
    # Also filter by context length
    def filter_by_length(x):
        ctx_len = x.get("context_len", len(x.get("context_window_text", "")))
        return min_context_len <= ctx_len <= max_context_len
    
    try:
        filtered = filtered.filter(filter_by_length)
        print(f"After context length filter: {len(filtered)} samples")
    except:
        pass
    
    # Take samples
    samples = list(filtered)[:num_samples]
    print(f"Using {len(samples)} samples for experiment")
    
    if len(samples) == 0:
        print("ERROR: No samples found! Check the dataset_filter value.")
        return None
    
    # Setup RLM
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    model_name = os.getenv("MODEL_NAME", "deepseek-chat")
    
    if not api_key:
        raise ValueError("No API key found!")
    
    use_deepseek = "deepseek" in model_name.lower()
    backend_kwargs = {"api_key": api_key, "model_name": model_name}
    if use_deepseek:
        backend_kwargs["base_url"] = base_url
    
    print(f"Model: {model_name}")
    
    # Create RLM
    logger = RLMLogger()
    rlm = RLM(
        backend="openai",
        backend_kwargs=backend_kwargs,
        max_depth=max_depth,
        max_iterations=10,
        logger=logger,
        verbose=True,
    )
    
    # Run experiment
    results: list[dict] = []
    total_score = 0.0
    correct_count = 0
    # For cost / token analysis (Figure 3 style)
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    
    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i+1}/{len(samples)} ---")
        
        # Get data
        context = sample.get("context_window_text", sample.get("context", ""))
        question = sample.get("question", "")
        gold_answer = sample.get("answer", "")
        answer_type = sample.get("answer_type", None)
        dataset_name = sample.get("dataset", "unknown")
        
        print(f"Dataset: {dataset_name}")
        print(f"Question: {question[:100]}...")
        print(f"Gold: {gold_answer[:50] if isinstance(gold_answer, str) else gold_answer}...")
        
        try:
            # RLM completion
            result = rlm.completion(
                prompt=context,
                root_prompt=question
            )
            
            response = result.response
            eval_result = evaluate_oolong_response(response, gold_answer, answer_type)
            
            total_score += eval_result["score"]
            if eval_result["correct"]:
                correct_count += 1
                print(f"✓ CORRECT: {eval_result['extracted_answer']}")
            else:
                print(f"✗ Score: {eval_result['score']:.2f}")
                print(f"  Extracted: {eval_result['extracted_answer'][:100]}")
                print(f"  Response: {response[:150]}...")
            # Token / cost usage
            usage_info = {}
            if result.usage_summary and result.usage_summary.model_usage_summaries:
                summary = next(iter(result.usage_summary.model_usage_summaries.values()))
                input_tokens = getattr(summary, "total_input_tokens", 0) or 0
                output_tokens = getattr(summary, "total_output_tokens", 0) or 0
                cost = getattr(summary, "total_cost", 0.0) or 0.0
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_cost += float(cost)
                usage_info = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "cost": float(cost),
                }

            results.append({
                "sample_id": i,
                "dataset": dataset_name,
                "question": question,
                "gold_answer": str(gold_answer),
                "rlm_response": response,
                "score": eval_result["score"],
                "correct": eval_result["correct"],
                "extracted_answer": eval_result["extracted_answer"],
                "parse_confidence": eval_result["parse_confidence"],
                "execution_time": result.execution_time,
                "usage": usage_info,
            })
            
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "sample_id": i,
                "dataset": dataset_name,
                "question": question,
                "gold_answer": str(gold_answer),
                "rlm_response": f"ERROR: {str(e)}",
                "score": 0.0,
                "correct": False,
                "extracted_answer": "",
                "parse_confidence": "error",
                "execution_time": 0,
            })
    
    # Calculate metrics
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
              f"~{total_tokens} total; Approx cost: ${total_cost:.4f}")
    
    # Save results
    output = {
        "experiment": output_name,
        "model": model_name,
        "max_depth": max_depth,
        "dataset_filter": dataset_filter,
        "num_samples": len(samples),
        "accuracy": accuracy,
        "avg_score": avg_score,
        "correct_count": correct_count,
        "avg_execution_time": avg_time,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost,
        "results": results,
    }
    
    output_path = Path(f"results/{output_name}_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    rlm.close()
    return output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OOLONG Experiment with RLM")
    parser.add_argument("--depth", type=int, default=1, help="RLM max_depth")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples (paper uses 50)")
    parser.add_argument("--dataset", type=str, default="trec_coarse", help="Dataset filter (paper uses trec_coarse)")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    args = parser.parse_args()
    
    name = args.name or f"oolong_depth{args.depth}"
    
    run_oolong_experiment(
        max_depth=args.depth,
        num_samples=args.samples,
        output_name=name,
        dataset_filter=args.dataset
    )
