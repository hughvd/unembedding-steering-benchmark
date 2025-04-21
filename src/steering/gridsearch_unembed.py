# # Setting up the environment
# from init_env import set_source_root
# project_root = set_source_root()

# Imports
from src.models.gemma_loader import load_gemma
from src.steering.unembedding_steering import get_unembedding_vector
from positivity_eval import evaluate_positivity
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import torch

# Load model and tokenizer
model, tokenizer = load_gemma(model_name="google/gemma-2-2b")

# Load eval dataset
with open("data/positive_sentiment_eval_dataset.json", "r") as f:
   dataset = json.load(f)

# Grid search parameters
layers = np.arange(model.config.num_hidden_layers)
scale_factors = np.arange(0.1, 1.1, 0.1)

# Define steering tokens
token_list = [[" happy"],
              [" happy", " amazing", " splendid", " incredible", " joyful"],
              [" happy", " amazing", " splendid", " incredible", " joyful",
               " delighted", " excited", " thrilled", " ecstatic", " overjoyed"],
              [" happy", " amazing", " splendid", " incredible", " joyful",
               " delighted", " excited", " thrilled", " ecstatic", " overjoyed", 
               " euphoric", " jubilant", " blissful", " cheerful", " content",
               " satisfied", " pleased", " gratified", " fulfilled", " fabulous"] 
            ]


for steering_tokens in token_list:
    # Get steering vector
    steering_vector = get_unembedding_vector(
        model=model,
        tokenizer=tokenizer,
        steering_tokens=steering_tokens,
        combine_method="mean"
    )

    results = []

    for l in tqdm(layers):
        for s in tqdm(scale_factors):
            with torch.no_grad():
                # Hook the model, scaling by multiple of residual stream norm
                def hook(module, input, output):
                    residual_stream_norm = torch.norm(output[0][:, -1])
                    output[0][:, -1] += s * residual_stream_norm * steering_vector
                    return output[0],

                # Register a forward hook on the specified layer
                curr_hook = model.model.layers[l].register_forward_hook(hook)

                # Evaluate positivity
                result = evaluate_positivity(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    device="cuda"
                )

                results.append({
                    "layer": int(l),
                    "scale": float(s),
                    "avg_score": result["avg_score"]
                })

                # Unhook the model
                curr_hook.remove()

    # Turn into DataFrame
    df = pd.DataFrame(results)
    n = len(steering_tokens)
    # Save to CSV
    df.to_csv(f"{n}_grid_search_results.csv", index=False)
