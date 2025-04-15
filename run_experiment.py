import os
import json
import torch
import wandb
import numpy as np
from tqdm import tqdm

from src.models.gemma_loader import load_gemma
from src.unembedding_steering import get_unembedding_vector
from src.positivity_eval import evaluate_positivity

# Load learned steering vectors
C_lp = torch.tensor(np.load("outputs/C_lp.npy"))
C_perp = torch.tensor(np.load("outputs/C_perp.npy"))
C_caa = torch.tensor(np.load("outputs/CAA_sae.npy"))

# Define steering vectors dictionary
STEERING_VECTORS = {
    "C_lp": C_lp,
    "C_perp": C_perp,
    "C_caa": C_caa,
    "W_pos": get_unembedding_vector(model, tokenizer, ["happy", "amazing", "splendid", "incredible"])
}

def apply_steering_hook(model, layer, vector, scale):
    """Apply a forward hook that steers residual stream with given vector."""
    def hook_fn(module, input, output):
        output[0][:, -1] += scale * vector.to(output[0].device)
        return output
    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    return handle

def main():
    # Initialize wandb
    wandb.init(project="steering_experiments", config={
        "layer": 15,
        "scale": 60,
        "eval_method": "positivity_logit"
    })
    
    # Load model/tokenizer
    model, tokenizer = load_gemma(model_name="google/gemma-2-9b-it", device="cuda")
    
    # Load prompt dataset
    with open("data/positive_sentiment_eval_dataset.json", "r") as f:
        prompt_dataset = json.load(f)

    # Parameters
    LAYER = wandb.config["layer"]
    SCALE = wandb.config["scale"]

    for name, vector in STEERING_VECTORS.items():
        print(f"Evaluating {name}...")
        handle = apply_steering_hook(model, LAYER, vector / vector.norm(), SCALE)

        results = evaluate_positivity(model, tokenizer, prompt_dataset)
        wandb.log({
            f"{name}/avg_score": results["avg_score"],
            f"{name}/scores": results["individual_scores"]
        })

        handle.remove()

if __name__ == "__main__":
    main()
