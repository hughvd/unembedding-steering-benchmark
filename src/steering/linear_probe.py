#!/usr/bin/env python3

"""
linear_probe.py

This script learns the linear probe (C_lp) vector for a selected concept (e.g., positive sentiment)
by training a logistic regression classifier on hidden representations extracted from SST-2.
The script demonstrates both raw hidden state extraction and SAE-encoded hidden state extraction.

File structure:
This file should live in src/steering/.

It uses the gemma model loader from src/models/gemma_loader.py.
It uses the SAE setup (as in cum_sae.py) to obtain sparse encodings of the hidden states.

References:
- Alain, G., & Bengio, Y. (2016). Understanding intermediate layers using linear probes.
- Rimsky et al. (2024). Steering Llama 2 via Contrastive Activation Addition.
- Park et al. (2023). The Linear Representation Hypothesis and the Geometry of Large Language Models.
"""

import os
import random
import numpy as np
import torch
from src.steering.hooks import extract_layer_features, decode_sae_direction
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import the Gemma loader from the models folder
from src.models.gemma_loader import load_gemma

# Import SAE from sae_lens if available
try:
    from sae_lens import SAE
except ImportError:
    raise ImportError("SAE module is required. Please install sae_lens or ensure its path is available.")

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Configuration
MODEL_NAME = "google/gemma-2-9b-it"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_0/width_16k/canonical"
NUM_SAMPLES = 200
LAYER_RAW = -1
LAYER_SAE = 0

# Load Gemma model & tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = load_gemma(model_name=MODEL_NAME, device=device)
model.eval()

# Load the SAE for encoding hidden states
sae, sae_cfg, sparsity = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID)
print("Loaded SAE with config:", sae_cfg, "sparsity:", sparsity)

# Load the SST-2 dataset
dataset = load_dataset("glue", "sst2")
train_sentences = dataset['train']['sentence'][:NUM_SAMPLES]
train_labels = dataset['train']['label'][:NUM_SAMPLES]
print(f"Using {len(train_sentences)} samples from SST-2 for training the probe.")

# Hidden state extraction functions
def extract_raw_hidden_state(text, layer_idx=LAYER_RAW):
    """Extract the raw hidden state from the model given a text input."""
    return extract_layer_features(model, tokenizer, text, layer_idx).numpy()

def extract_hidden_state_with_sae(text, layer_idx=LAYER_SAE):
    """Extract hidden states, then encode them with the SAE."""
    return extract_layer_features(model, tokenizer, text, layer_idx, sae=sae).numpy()

# Extract representations
raw_reps = []
sae_reps = []

for txt in train_sentences:
    try:
        raw_rep = extract_raw_hidden_state(txt)
        raw_reps.append(raw_rep)
    except Exception as e:
        print("Error extracting raw hidden state for text:", txt, e)
    
    try:
        sae_rep = extract_hidden_state_with_sae(txt)
        sae_reps.append(sae_rep)
    except Exception as e:
        print("Error extracting SAE hidden state for text:", txt, e)

raw_reps = np.array(raw_reps)
sae_reps = np.array(sae_reps)
labels = np.array(train_labels)

print("Raw hidden states shape:", raw_reps.shape)
print("SAE-encoded hidden states shape:", sae_reps.shape)

# Train Linear Probe
print("\nTraining linear probe on raw hidden states...")
clf_raw = LogisticRegression(max_iter=1000)
clf_raw.fit(raw_reps, labels)
C_raw = clf_raw.coef_.flatten()
print("Learned raw linear probe vector shape:", C_raw.shape)

print("\nTraining linear probe on SAE-encoded hidden states...")
clf_sae = LogisticRegression(max_iter=1000)
clf_sae.fit(sae_reps, labels)
C_sae = clf_sae.coef_.flatten()
print("Learned SAE linear probe vector shape:", C_sae.shape)

# Final output vector
C_lp = decode_sae_direction(torch.as_tensor(C_sae, dtype=torch.float32), sae.W_dec).numpy()
print("\nFinal learned linear probe (C_lp) vector (using SAE encoding) has shape:", C_lp.shape)
print("C_lp vector (first 10 elements):", C_lp[:10])

# Save the learned vector
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, "C_lp.npy"), C_lp)
np.save(os.path.join(output_dir, "C_lp_sae.npy"), C_sae)
print(f"Saved C_lp vector to {os.path.join(output_dir, 'C_lp.npy')}")
