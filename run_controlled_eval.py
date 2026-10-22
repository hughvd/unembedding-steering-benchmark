"""Small counterbalanced unembedding intervention check with retained raw scores.

This uses a preset layer and scales, not a search for the best evaluation score.
Example:
  python run_controlled_eval.py --model Qwen/Qwen2.5-3B-Instruct --local-only
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from positivity_eval import evaluate_positivity
from src.steering.hooks import steer_residual


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--output", type=Path, default=root / "data/counterbalanced_results.json")
    args = parser.parse_args()
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        local_files_only=args.local_only,
    ).eval()
    path = root / "data/positive_sentiment_eval_dataset.json"
    dataset = json.loads(path.read_text(encoding="utf-8"))
    tokens = [" good", " great", " happy"]
    encoded = [tokenizer.encode(token, add_special_tokens=False) for token in tokens]
    if any(len(ids) != 1 for ids in encoded):
        raise ValueError("The preset vocabulary must contain single-token strings")
    token_ids = [ids[0] for ids in encoded]
    direction = model.get_output_embeddings().weight[token_ids].detach().float().mean(dim=0)
    layer = 3 * len(model.model.layers) // 4
    scales = [0.0, 0.1, 0.5]
    report = {
        "model": args.model, "model_revision": getattr(model.config, "_commit_hash", None),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "dtype": str(dtype), "seed": 42, "n_prompts": len(dataset),
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "protocol": "counterbalanced_choice_logits_v2",
        "selection": "preset layer floor(3L/4), preset scales; no best-cell selection",
        "layer": layer, "scales": scales, "tokens": tokens, "token_ids": token_ids,
        "intervention": "all prompt tokens; scale times each example's last-token residual norm",
        "results": [],
    }
    baseline = None
    for scale in scales:
        handle = model.model.layers[layer].register_forward_hook(
            lambda module, inputs, output: steer_residual(output, direction, scale)
        )
        try:
            result = evaluate_positivity(model, tokenizer, dataset, device=str(model.device))
        finally:
            handle.remove()
        if baseline is None:
            baseline = np.asarray(result["individual_scores"])
        result["scale"] = scale
        result["mean_paired_change_from_unsteered"] = float(
            np.mean(np.asarray(result["individual_scores"]) - baseline)
        )
        report["results"].append(result)
        print(f"scale={scale}: mean={result['avg_score']:.6f}, paired change={result['mean_paired_change_from_unsteered']:.6f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
