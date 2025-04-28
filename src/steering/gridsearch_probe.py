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

# Load linear probe steering vectors (n_layers x n_behaviors x d_model)
sentiment_tensor_linear = torch.load("src/steering/toy_mov_probes.pth")
sentiment_tensor_perp = torch.load("src/steering/perp_toy_mov_probes.pth")

# Grid search parameters
layers = np.arange(model.config.num_hidden_layers)
scale_factors = np.arange(0.1, 1.1, 0.1)

k = 0
for tensor in [sentiment_tensor_linear, sentiment_tensor_perp]:
    # 0 = positivity, 1 = negativity
    for i in [0, 1]:
        results = []

        # Grid search
        for l in tqdm(layers):
            # Get steering vector
            steering_vector = tensor[l][i, :].unsqueeze(0).to("cuda")
            # Normalize the steering vector
            steering_vector /= torch.norm(steering_vector)

            for s in tqdm(scale_factors):
                with torch.no_grad():
                    # Hook the model, scaling by multiple of residual stream norm
                    def hook(module, input, output):
                        residual_stream_norm = torch.norm(output[0][:, -1])
                        output[0][:, :] += s * residual_stream_norm * steering_vector
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
        # Save to CSV
        if k == 0:
            df.to_csv(f"toy_{i}_grid_search_results.csv", index=False)
        else:
            df.to_csv(f"perp_toy_{i}_grid_search_results.csv", index=False)
    k += 1