"""
RULER S-NIAH Experiment with RLM

This script:
1. Uses RULER's official data generation to create S-NIAH samples
2. Runs RLM on the samples
3. Evaluates using RULER's metric

Reference: https://github.com/NVIDIA/RULER
"""

import os
import sys
import json
import re
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add RULER scripts to path (official RULER repo)
RULER_PATH = Path(__file__).parent.parent / "RULER" / "scripts"
sys.path.insert(0, str(RULER_PATH))


def generate_ruler_data_official(
    output_dir: str,
    max_seq_length: int = 4096,
    num_samples: int = 50,
    task: str = "niah_single_2",
    tokenizer_type: str = "openai",
    tokenizer_path: str = "cl100k_base",
) -> Path:
    """
    Generate RULER data using official scripts (exact RULER S-NIAH).

    We call RULER's `prepare.py` with:
    - benchmark = synthetic
    - task      = niah_single_2  (essay haystack, single numeric needle)
    - num_samples = 50           (as in RLM paper: 50 single tasks)

    IMPORTANT: For perfect comparability with the paper and RULER,
    we DO NOT fall back to a custom generator here. Any failure should
    be treated as a configuration issue to fix (tokenizer, NLTK, paths, etc.).
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists in RULER's expected location
    data_file = output_path / f"{task}" / "validation.jsonl"
    if data_file.exists():
        print(f"Data already exists at {data_file}")
        return data_file
    
    # Call RULER's prepare.py
    prepare_script = RULER_PATH / "data" / "prepare.py"
    
    if not prepare_script.exists():
        raise FileNotFoundError(f"RULER prepare.py not found at {prepare_script}")
    
    cmd = [
        "python",
        str(prepare_script),
        "--save_dir",
        str(output_path),
        "--benchmark",
        "synthetic",
        "--task",
        task,
        "--tokenizer_path",
        tokenizer_path,
        "--tokenizer_type",
        tokenizer_type,
        "--max_seq_length",
        str(max_seq_length),
        "--num_samples",
        str(num_samples),
    ]

    print(f"Running official RULER generator: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # Do NOT silently fall back – this would break exact comparability.
        raise RuntimeError(
            "RULER data generation failed.\n"
            "Command: "
            + " ".join(cmd)
            + "\nStdout:\n"
            + result.stdout
            + "\nStderr:\n"
            + result.stderr
        )

    return data_file


def generate_ruler_data_manual(
    output_dir: str,
    max_seq_length: int = 4096,
    num_samples: int = 30
):
    """
    Fallback: Generate RULER-style S-NIAH data manually.
    
    This follows the same format as RULER's official generation.
    """
    import random
    
    print(f"Generating RULER S-NIAH data manually...")
    print(f"Context length: {max_seq_length} tokens")
    print(f"Samples: {num_samples}")
    
    # Word lists (simplified version of RULER's wonderwords)
    nouns = ["apple", "banana", "cherry", "dragon", "elephant", "falcon", 
             "giraffe", "horizon", "island", "jungle", "kangaroo", "lemon",
             "mountain", "night", "ocean", "penguin", "quantum", "rainbow"]
    adjs = ["magic", "special", "secret", "hidden", "mysterious", "ancient",
            "golden", "silver", "crystal", "shadow", "bright", "dark"]
    
    words = [f"{adj}-{noun}" for adj in adjs for noun in nouns]
    
    # Filler sentences
    filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    
    samples = []
    random.seed(42)
    
    for i in range(num_samples):
        # Generate needle
        key = random.choice(words)
        value = str(random.randint(1000000, 9999999))  # 7-digit number
        needle = f"One of the special magic numbers for {key} is: {value}."
        
        # Calculate approximate context size
        # Target: max_seq_length tokens, but leave room for query
        target_chars = (max_seq_length - 50) * 4  # Approximate chars
        
        # Generate haystack
        haystack = filler * (target_chars // len(filler) + 1)
        
        # Insert needle at random position
        haystack_words = haystack.split()
        insert_pos = random.randint(0, len(haystack_words))
        haystack_words.insert(insert_pos, needle)
        context = " ".join(haystack_words)[:target_chars]
        
        # Create sample in RULER format
        template = """Some special magic numbers are hidden within the following text. Make sure to memorize it. I will quiz you about the numbers afterwards.
{context}
What are all the special magic numbers for {query} mentioned in the provided text? The special magic numbers for {query} mentioned in the provided text are"""
        
        input_text = template.format(context=context, query=key)
        
        samples.append({
            "index": i,
            "input": input_text,
            "outputs": [value],
            "needle_key": key,
            "needle_value": value,
            "length": max_seq_length
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_samples}")
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_file = output_path / "ruler_sniah_manual.json"
    
    with open(data_file, "w") as f:
        json.dump(samples, f, indent=2)
    
    print(f"Saved {num_samples} samples to {data_file}")
    return data_file


def load_ruler_data(data_file: str):
    """Load RULER data from jsonl or json file."""
    data_file = Path(data_file)
    
    if data_file.suffix == ".jsonl":
        samples = []
        with open(data_file) as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples
    else:
        with open(data_file) as f:
            return json.load(f)


def evaluate_ruler_response(response: str, gold_answers: list) -> dict:
    """
    Evaluate RULER response.
    
    RULER metric: Check if all gold answers appear in response.
    """
    response = response.strip().lower()
    
    found = []
    for gold in gold_answers:
        gold_str = str(gold).lower()
        if gold_str in response:
            found.append(gold)
    
    # Calculate score
    if len(gold_answers) == 0:
        score = 0.0
    else:
        score = len(found) / len(gold_answers)
    
    return {
        "score": score,
        "found": found,
        "total": len(gold_answers),
        "correct": len(found) == len(gold_answers)
    }


def run_ruler_rlm_experiment(
    max_depth: int = 1,
    num_samples: int = 50,  # Paper uses 50 samples
    max_seq_length: int = 4096,
    output_name: str = "ruler_depth1"
):
    """
    Run RULER experiment with RLM.
    """
    from rlm import RLM
    from rlm.logger import RLMLogger
    
    print("=" * 60)
    print(f"RULER S-NIAH Experiment with RLM")
    print(f"max_depth={max_depth}, samples={num_samples}, length={max_seq_length}")
    print("=" * 60)
    
    # Generate or load data
    data_dir = Path("data/ruler")
    try:
        data_file = generate_ruler_data_official(
            output_dir=str(data_dir),
            max_seq_length=max_seq_length,
            num_samples=num_samples,
            tokenizer_type="openai",
            tokenizer_path="cl100k_base"
        )
    except Exception as e:
        print(f"Official generation failed: {e}")
        print("Using manual generation...")
        data_file = generate_ruler_data_manual(
            output_dir=str(data_dir),
            max_seq_length=max_seq_length,
            num_samples=num_samples
        )
    
    samples = load_ruler_data(data_file)
    samples = samples[:num_samples]
    print(f"Loaded {len(samples)} samples")
    
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
    # For cost / token plots (Figure 3 风格)
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    
    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i+1}/{len(samples)} ---")
        
        # Get data
        input_text = sample.get("input", "")
        outputs = sample.get("outputs", sample.get("answer", [""]))
        if isinstance(outputs, str):
            outputs = [outputs]
        
        # Extract key info if available
        key = sample.get("needle_key", "unknown")
        value = sample.get("needle_value", outputs[0] if outputs else "")
        print(f"Key: {key}, Expected: {value}")
        
        try:
            # RLM expects: prompt (context), root_prompt (query)
            # We need to split the input into context and query
            # RULER format has context then query in the template
            
            # For RLM, we pass the full input as context
            # and use a simple query prompt
            query = f"What are all the special magic numbers for {key} mentioned in the provided text?"
            
            result = rlm.completion(
                prompt=input_text,  # Full context
                root_prompt=query   # Query
            )
            
            response = result.response
            eval_result = evaluate_ruler_response(response, outputs)
            
            total_score += eval_result["score"]
            if eval_result["correct"]:
                correct_count += 1
                print(f"✓ CORRECT: {eval_result['found']}")
            else:
                print(f"✗ PARTIAL: found {eval_result['found']} / expected {outputs}")
                print(f"  Response: {response[:150]}...")
            # Token / cost usage (for later cost plots)
            usage_info = {}
            if result.usage_summary and result.usage_summary.model_usage_summaries:
                # 取第一个模型的统计（这里只有一个 DeepSeek 模型）
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
                "needle_key": key,
                "gold_answers": outputs,
                "rlm_response": response,
                "score": eval_result["score"],
                "correct": eval_result["correct"],
                "found": eval_result["found"],
                "execution_time": result.execution_time,
                "usage": usage_info,
            })
            
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "sample_id": i,
                "needle_key": key,
                "gold_answers": outputs,
                "rlm_response": f"ERROR: {str(e)}",
                "score": 0.0,
                "correct": False,
                "found": [],
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
        "max_seq_length": max_seq_length,
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
    
    parser = argparse.ArgumentParser(description="RULER S-NIAH Experiment with RLM")
    parser.add_argument("--depth", type=int, default=1, help="RLM max_depth")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples")
    parser.add_argument("--length", type=int, default=4096, help="Context length in tokens")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    args = parser.parse_args()
    
    name = args.name or f"ruler_depth{args.depth}"
    
    run_ruler_rlm_experiment(
        max_depth=args.depth,
        num_samples=args.samples,
        max_seq_length=args.length,
        output_name=name
    )
