#!/usr/bin/env python3
"""
c_perp.py

This script computes steering vectors that are constrained to be orthogonal to the 
aggregated positive token unembedding vector, W_pos_mean. In our proposal, we learn a 
concept vector (for positive sentiment) via a linear probe and then obtain the “orthogonal” 
variant (C_perp) by subtracting its projection onto the unembedding direction.

Specifically, if C is the learned concept vector (from a logistic regression classifier) and 
W is the aggregated positive unembedding (e.g. mean of tokens: "positive", "good", "great", "amazing", "excellent"),
then:

    C_perp = C - ( (C·W) / ||W||² ) * W

This file computes C_perp for both raw and SAE-encoded representations using the SST-2 dataset, 
using the same procedure as in our previous files. The resulting vectors are saved to disk and 
their cosine similarities with W_pos_mean are printed for verification.

File Structure:
  - This file should reside in src/steering/.
  - It loads the Gemma model and tokenizer using load_gemma from src/models/gemma_loader.py.
  - It uses the SAE component from sae_lens (as in cum_sae.py).
  - It computes both raw and SAE versions of C (from a logistic regression probe on SST-2), then 
    produces C_perp_raw and C_perp_sae respectively.

References:
  - Panickssery et al. (2023). Steering Llama 2 via Contrastive Activation Addition.
  - Alain & Bengio (2016). Understanding Intermediate Layers Using Linear Probes.
  - Park, Choe, & Veitch (2023). The Linear Representation Hypothesis and the Geometry of Large Language Models.
"""

import numpy as np
import torch
import random
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Import the Gemma model loader
from src.models.gemma_loader import load_gemma

# Import SAE from sae_lens
try:
    from sae_lens import SAE
except ImportError:
    raise ImportError("SAE module is required. Please install sae_lens or ensure the module is on the PYTHONPATH.")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# --- Configuration ---
MODEL_NAME = "google/gemma-2-9b-it"  # Gemma model variant from gemma_loader
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_0/width_16k/canonical"
NUM_SAMPLES = 200  # Number of SST-2 examples to use
LAYER_RAW = -1     # Use the last layer for raw hidden states
LAYER_SAE = 0      # Use layer 0 for SAE encoding
POSITIVE_TOKENS = ["positive", "good", "great", "amazing", "excellent"]

# --- Load Model and Tokenizer ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = load_gemma(model_name=MODEL_NAME, device=device)
model.eval()

# --- Load SAE ---
sae, sae_cfg, sparsity = SAE.from_pretrained(
    release=SAE_RELEASE,
    sae_id=SAE_ID,
)
print("Loaded SAE with config:", sae_cfg, "sparsity:", sparsity)

# --- Data Preparation ---
dataset = load_dataset("glue", "sst2")
train_sentences = dataset['train']['sentence'][:NUM_SAMPLES]
train_labels = dataset['train']['label'][:NUM_SAMPLES]
print(f"Using {len(train_sentences)} samples from SST-2 for probing.")

# --- Representation Extraction Functions ---
def extract_raw_hidden_state(text, layer_idx=LAYER_RAW):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(next(model.parameters()).device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Average over token sequence dimension
    hidden = outputs.hidden_states[layer_idx]  # (1, seq_len, hidden_dim)
    rep = hidden.mean(dim=1).squeeze()         # (hidden_dim,)
    return rep.cpu().numpy()

def extract_hidden_state_with_sae(text, layer_idx=LAYER_SAE):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(next(model.parameters()).device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer_idx]  # (1, seq_len, hidden_dim)
    sae_out = sae.encode(hidden)               # (1, seq_len, hidden_dim)
    rep = sae_out.mean(dim=1).squeeze()
    return rep.cpu().numpy()

# --- Extract Representations from SST-2 ---
raw_reps = []
sae_reps = []
for txt in train_sentences:
    try:
        raw_reps.append(extract_raw_hidden_state(txt))
    except Exception as e:
        print("Error extracting raw hidden state for:", txt, e)
    try:
        sae_reps.append(extract_hidden_state_with_sae(txt))
    except Exception as e:
        print("Error extracting SAE hidden state for:", txt, e)

raw_reps = np.array(raw_reps)
sae_reps = np.array(sae_reps)
labels = np.array(train_labels)
print("Raw hidden states shape:", raw_reps.shape)
print("SAE-encoded hidden states shape:", sae_reps.shape)

# --- Train Logistic Regression to Get C (Concept Vector) ---
print("\nTraining linear probe on raw hidden states...")
clf_raw = LogisticRegression(max_iter=1000)
clf_raw.fit(raw_reps, labels)
C_raw = clf_raw.coef_.flatten()
print("Learned raw concept vector shape:", C_raw.shape)

print("\nTraining linear probe on SAE-encoded hidden states...")
clf_sae = LogisticRegression(max_iter=1000)
clf_sae.fit(sae_reps, labels)
C_sae = clf_sae.coef_.flatten()
print("Learned SAE concept vector shape:", C_sae.shape)

# --- Compute Aggregated Positive Token Unembedding (W_pos_mean) ---
pos_token_vectors = []
for token in POSITIVE_TOKENS:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    tok_id = token_ids[0]
    vec = model.transformer.wte.weight[tok_id].detach().cpu().numpy()
    pos_token_vectors.append(vec)
pos_token_vectors = np.stack(pos_token_vectors)
W_pos_mean = pos_token_vectors.mean(axis=0)
print("Aggregated positive token unembedding vector shape:", W_pos_mean.shape)

# --- Compute Orthogonal Projection for C_perp ---
def compute_C_perp(C, W):
    """Compute the component of C orthogonal to vector W."""
    proj = (np.dot(C, W) / np.dot(W, W)) * W
    C_perp = C - proj
    return C_perp

C_perp_raw = compute_C_perp(C_raw, W_pos_mean)
C_perp_sae = compute_C_perp(C_sae, W_pos_mean)

# --- Check Orthogonality via Cosine Similarity ---
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

cos_sim_raw = cosine_similarity(C_perp_raw, W_pos_mean)
cos_sim_sae = cosine_similarity(C_perp_sae, W_pos_mean)
print("\nCosine similarity between C_perp_raw and W_pos_mean:", cos_sim_raw)
print("Cosine similarity between C_perp_sae and W_pos_mean:", cos_sim_sae)

# Ideally these values should be near zero, showing effective orthogonalization.

# --- Save C_perp vectors ---
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, "C_perp_raw.npy"), C_perp_raw)
np.save(os.path.join(output_dir, "C_perp_sae.npy"), C_perp_sae)
print(f"\nSaved C_perp_raw to {os.path.join(output_dir, 'C_perp_raw.npy')}")
print(f"Saved C_perp_sae to {os.path.join(output_dir, 'C_perp_sae.npy')}")

# End of script.
