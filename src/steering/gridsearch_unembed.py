from src.models.gemma_loader import load_gemma
from src.steering.unembedding_steering import get_unembedding_vector
from positivity_eval import evaluate_positivity
import numpy as np
import torch

# Load model and tokenizer
model, tokenizer = load_gemma(model_name="google/gemma-2-2b")

# Grid search parameters
layers = np.arange(model.config.num_hidden_layers)
scale_factors = np.arange(0.0, 1.1, 0.1)

# Define steering tokens

