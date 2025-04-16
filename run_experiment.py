#!/usr/bin/env python3

import os
import json
import argparse
import wandb
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.models.gemma_loader import load_gemma

'''
Usage:
python run_experiment.py \
  --method C_perp \
  --prompt_file prompts/sentiment_eval.json \
  --scale 1.5 \
  --layer 15

'''

# -----------------------
# Steering Vector Loader
# -----------------------
STEERING_VECTOR_PATHS = {
    "C_lp": "outputs/C_lp.npy",
    "C_perp": "outputs/C_perp.npy",
    "CAA": "outputs/CAA_sae.npy",
    "W_pos": "outputs/W_pos.npy"  
}

def load_steering_vector(method):
    path = STEERING_VECTOR_PATHS.get(method)
    if path is None or not os.path.exists(path):
        raise ValueError(f"No saved vector found for method: {method}")
    return torch.tensor(np.load(path), dtype=torch.float32)


# -----------------------
# Evaluation Core
# -----------------------
@torch.no_grad()
def evaluate_prompt(model, tokenizer, prompt, option_a, option_b, steering_vec, layer, scale):
    """
    Computes logit difference (A - B) under steering vector injection.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    a_id = tokenizer.encode(option_a, add_special_tokens=False)[0]
    b_id = tokenizer.encode(option_b, add_special_tokens=False)[0]

    seq_len = inputs.input_ids.size(1)

    def hook_fn(module, input, output):
        output[0][:, -1] += scale * steering_vec.to(output[0].device)
        return output

    hook = model.model.layers[layer].register_forward_hook(hook_fn)
    logits = model(**inputs).logits
    hook.remove()

    last_logits = logits[0, -1]
    logit_diff = last_logits[a_id].item() - last_logits[b_id].item()

    return logit_diff


# -----------------------
# Main Experiment Loop
# -----------------------
def run_experiment(config):
    # Init wandb
    wandb.init(project="steering-benchmark", config=config)

    # Load model/tokenizer
    model, tokenizer = load_gemma(model_name=config["model_name"], device="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Load steering vector
    steering_vec = load_steering_vector(config["method"])
    steering_vec = steering_vec / steering_vec.norm()  # normalize

    # Load prompts
    with open(config["prompt_file"], "r") as f:
        prompt_data = json.load(f)

    # Evaluate each prompt
    results = []
    for entry in tqdm(prompt_data, desc="Evaluating prompts"):
        prompt = entry["prompt"]
        a = entry["A"]
        b = entry["B"]
        logit_diff = evaluate_prompt(model, tokenizer, prompt, a, b, steering_vec, config["layer"], config["scale"])

        wandb.log({
            "prompt": prompt,
            "option_A": a,
            "option_B": b,
            "logit_diff": logit_diff,
            "method": config["method"],
            "scale": config["scale"],
            "layer": config["layer"],
        })

        results.append({
            "prompt": prompt,
            "A": a,
            "B": b,
            "logit_diff": logit_diff
        })

    return results


# -----------------------
# Entry Point + CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run steering experiment across prompts.")
    parser.add_argument("--method", type=str, choices=["C_lp", "C_perp", "CAA", "W_pos"], required=True,
                        help="Which steering method to use.")
    parser.add_argument("--prompt_file", type=str, required=True,
                        help="Path to prompt file (in JSON format).")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scaling factor to apply to steering vector.")
    parser.add_argument("--layer", type=int, default=15,
                        help="Layer to apply steering at.")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it",
                        help="Model name (HuggingFace identifier).")

    args = parser.parse_args()
    config = vars(args)
    run_experiment(config)
