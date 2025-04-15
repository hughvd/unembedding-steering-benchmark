#!/usr/bin/env python3

"""
caa.py

This script computes candidate Concept Activation Attribution (CAA) vectors for a chosen concept
(here positive sentiment) by contrasting representations of positive and negative examples from
the SST-2 dataset. It computes both raw and SAE-encoded hidden state versions.

CAA = avg(h_positive) - avg(h_negative)

Inspired by:
- Panickssery et al. (2023). Steering Llama 2 via Contrastive Activation Addition.
- Alain & Bengio (2016). Understanding intermediate layers using linear probes.
- Park et al. (2023). The Linear Representation Hypothesis and the Geometry of LLMs.
"""

import os
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA

# Import the Gemma model loader
from src.models.gemma_loader import load_gemma

# Import SAE
try:
    from sae_lens import SAE
except ImportError:
    raise ImportError("SAE module is required. Please install sae_lens or adjust your PYTHONPATH.")

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

POSITIVE_TOKENS = ["positive", "good", "great", "amazing", "excellent"]

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = load_gemma(model_name=MODEL_NAME, device=device)
model.eval()

# Load SAE
sae, sae_cfg, sparsity = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID)
print("Loaded SAE with config:", sae_cfg, "sparsity:", sparsity)

# Load SST-2 dataset
dataset = load_dataset("glue", "sst2")
train_sentences = dataset['train']['sentence'][:NUM_SAMPLES]
train_labels = dataset['train']['label'][:NUM_SAMPLES]
print(f"Using {len(train_sentences)} samples from SST-2.")

# Hidden state extraction functions
def extract_raw_hidden_state(text, layer_idx=LAYER_RAW):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer_idx]
    rep = hidden.mean(dim=1).squeeze()
    return rep.cpu().numpy()

def extract_hidden_state_with_sae(text, layer_idx=LAYER_SAE):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer_idx]
    sae_out = sae.encode(hidden)
    rep = sae_out.mean(dim=1).squeeze()
    return rep.cpu().numpy()

# Extract representations
raw_reps = []
sae_reps = []

for txt in train_sentences:
    try:
        raw_reps.append(extract_raw_hidden_state(txt))
    except Exception as e:
        print("Error extracting raw hidden state for text:", txt, e)
    try:
        sae_reps.append(extract_hidden_state_with_sae(txt))
    except Exception as e:
        print("Error extracting SAE hidden state for text:", txt, e)

raw_reps = np.array(raw_reps)
sae_reps = np.array(sae_reps)
labels = np.array(train_labels)

print("Raw hidden states shape:", raw_reps.shape)
print("SAE-encoded hidden states shape:", sae_reps.shape)

# Partition representations
pos_idx = labels == 1
neg_idx = labels == 0

raw_pos = raw_reps[pos_idx]
raw_neg = raw_reps[neg_idx]
sae_pos = sae_reps[pos_idx]
sae_neg = sae_reps[neg_idx]

print(f"Number of positive examples: {raw_pos.shape[0]}")
print(f"Number of negative examples: {raw_neg.shape[0]}")

# Compute CAA vectors
CAA_raw = raw_pos.mean(axis=0) - raw_neg.mean(axis=0)
CAA_sae = sae_pos.mean(axis=0) - sae_neg.mean(axis=0)

# Cosine similarity utility
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Positive token embeddings
pos_token_vectors = []
for token in POSITIVE_TOKENS:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    tok_id = token_ids[0]
    vec = model.transformer.wte.weight[tok_id].detach().cpu().numpy()
    pos_token_vectors.append(vec)

pos_token_vectors = np.stack(pos_token_vectors)
W_pos_mean = pos_token_vectors.mean(axis=0)

# First principal component of positive tokens
pca = PCA(n_components=1)
pca.fit(pos_token_vectors)
W_pos_pc1 = pca.components_[0]

# Compute similarities
cos_sim_raw_mean = cosine_similarity(CAA_raw, W_pos_mean)
cos_sim_raw_pc1 = cosine_similarity(CAA_raw, W_pos_pc1)
cos_sim_sae_mean = cosine_similarity(CAA_sae, W_pos_mean)
cos_sim_sae_pc1 = cosine_similarity(CAA_sae, W_pos_pc1)

print("\nCosine Similarities for CAA (raw representation):")
print("  CAA_raw & W_pos_mean:", cos_sim_raw_mean)
print("  CAA_raw & W_pos_pc1 :", cos_sim_raw_pc1)

print("\nCosine Similarities for CAA (SAE-encoded representation):")
print("  CAA_sae & W_pos_mean:", cos_sim_sae_mean)
print("  CAA_sae & W_pos_pc1 :", cos_sim_sae_pc1)

# Save CAA vectors
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, "CAA_raw.npy"), CAA_raw)
np.save(os.path.join(output_dir, "CAA_sae.npy"), CAA_sae)

print(f"\nSaved CAA_raw vector to {os.path.join(output_dir, 'CAA_raw.npy')}")
print(f"Saved CAA_sae vector to {os.path.join(output_dir, 'CAA_sae.npy')}")
