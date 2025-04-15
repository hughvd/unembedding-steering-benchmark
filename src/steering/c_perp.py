#!/usr/bin/env python3
"""
learn_c_perp.py

This script learns the steering vector C_perp for positive sentiment in the model's activation space,
subject to an orthogonality constraint with respect to the aggregated unembedding vector W_pos.
Instead of simply projecting C (learned from a probe) to be orthogonal, we modify the training loss so
that the learned vector directly minimizes positive classification error while being penalized for 
alignment with token-level unembeddings.

This approach is described in our project proposal and midterm report 
  
Procedure:
  1. Load Gemma-2-9b-it and its SAE encoder from the gemma_loader and cum_sae modules.
  2. Extract SAE-encoded hidden states from a subset of SST-2 from GLUE.
  3. Compute the aggregated positive token unembedding vector W_pos_mean from a list of positive sentiment tokens.
  4. Define a trainable parameter C_perp and optimize it using a loss function that includes:
       - A classification loss (binary cross-entropy loss applied to sigmoid(X·C_perp))
       - An orthogonality loss: lambda * ( (C_perp dot W_pos_mean)^2 )
  5. Save the learned C_perp vector.

References:
  - [1] Panickssery et al. Steering Llama 2 via Contrastive Activation Addition. arXiv:2312.xxxx, 2023.
  - [2] Midterm Report by Hugh Van Deventer et al.
  - [3] Representation engineering literature (e.g. Alain & Bengio, 2016; Park et al., 2023).

"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import model loader and SAE modules
from src.models.gemma_loader import load_gemma

# Import SAE from sae_lens; ensure its path is available.
try:
    from sae_lens import SAE
except ImportError:
    raise ImportError("SAE module is required. Please install sae_lens or include its path.")

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# --- Configuration Parameters ---
MODEL_NAME = "google/gemma-2-9b-it"   # Using Gemma-2-9b-it for experiments
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_0/width_16k/canonical"
NUM_SAMPLES = 500    # You can adjust the number of SST-2 samples for training
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
LAMBDA_ORTHO = 0.1   # Weight for the orthogonality penalty
LAYER_SAE = 0        # We use layer 0 for SAE encoding

# Define positive sentiment tokens for unembedding aggregation
POSITIVE_TOKENS = ["positive", "good", "great", "amazing", "excellent"]

# --- Load Model and SAE Encoder ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = load_gemma(model_name=MODEL_NAME, device=device)
model.eval()  # Set model to evaluation mode

# Load SAE encoder
sae, sae_cfg, sparsity = SAE.from_pretrained(
    release=SAE_RELEASE,
    sae_id=SAE_ID,
)
print("Loaded SAE: cfg:", sae_cfg, "sparsity:", sparsity)

# --- Load SST-2 Dataset ---
dataset = load_dataset("glue", "sst2")
# We'll use a larger subset for this experiment
train_sentences = dataset['train']['sentence'][:NUM_SAMPLES]
train_labels = dataset['train']['label'][:NUM_SAMPLES]  # labels: 0 (negative), 1 (positive)
print(f"Using {len(train_sentences)} samples from SST-2 for training.")

# --- Representation Extraction ---
def extract_sae_hidden_state(text):
    """
    Extract the SAE-encoded hidden state for the input text.
    Tokenize the input, get hidden states from layer LAYER_SAE, pass them through SAE encoder,
    and average over the token dimension.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[LAYER_SAE]  # shape: (1, seq_len, hidden_dim)
    sae_out = sae.encode(hidden)                # shape: (1, seq_len, hidden_dim)
    rep = sae_out.mean(dim=1).squeeze()           # shape: (hidden_dim,)
    return rep.cpu()  # return as CPU tensor

# Precompute hidden representations and labels
hidden_states_list = []
labels_list = []
for txt, lab in zip(train_sentences, train_labels):
    try:
        rep = extract_sae_hidden_state(txt)
        hidden_states_list.append(rep)
        labels_list.append(lab)
    except Exception as e:
        print(f"Error processing text: {txt} - {e}")

X = torch.stack(hidden_states_list)  # shape: (NUM_SAMPLES, hidden_dim)
y = torch.tensor(labels_list, dtype=torch.float32).unsqueeze(1)  # shape: (NUM_SAMPLES, 1)
print("Extracted hidden states shape:", X.shape)
print("Labels shape:", y.shape)

# --- Compute Aggregated Positive Token Unembedding Vector ---
def get_positive_unembedding(tokenizer, model, tokens):
    # Get token IDs and then retrieve from model's embedding matrix (assumes tied weights)
    token_vecs = []
    for tok in tokens:
        tok_id = tokenizer.encode(tok, add_special_tokens=False)[0]
        # Depending on model architecture: for GPT-2, typically use model.transformer.wte.weight
        vec = model.transformer.wte.weight[tok_id].detach().cpu()
        token_vecs.append(vec)
    token_vecs = torch.stack(token_vecs)  # shape: (num_tokens, hidden_dim)
    # Return the mean vector
    return token_vecs.mean(dim=0)

W_pos = get_positive_unembedding(tokenizer, model, POSITIVE_TOKENS)  # shape: (hidden_dim,)
print("Aggregated positive token unembedding vector shape:", W_pos.shape)
# Normalize for the penalty computation if needed
# (But we use dot product squared penalty directly)

# --- Define the C_perp Learner ---
class CPerpLearner(nn.Module):
    def __init__(self, hidden_dim):
        super(CPerpLearner, self).__init__()
        # Learn a weight vector (without bias) that will serve as the steering vector C_perp.
        # We initialize with small random values.
        self.C_perp = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        
    def forward(self, x):
        # x: shape (batch_size, hidden_dim)
        # Output logits: (batch_size, 1) (for binary classification)
        # Use a linear function (dot product with C_perp)
        logits = x @ self.C_perp.unsqueeze(1)  # shape: (batch_size, 1)
        return logits

# Instantiate the learner
hidden_dim = X.shape[1]
learner = CPerpLearner(hidden_dim).to(device)

# Define loss functions
bce_loss_fn = nn.BCEWithLogitsLoss()

def orthogonality_penalty(C, W):
    # Penalize the squared dot product between C and W.
    # Here, we do not strictly require normalization of C and W because the scale is learned; 
    # you may use squared cosine similarity if desired.
    dot = torch.dot(C, W.to(C.device))
    return dot ** 2

# --- Training Setup ---
optimizer = optim.Adam(learner.parameters(), lr=LEARNING_RATE)
num_batches = int(np.ceil(X.shape[0] / BATCH_SIZE))

print("Starting training of C_perp...")

learner.train()
for epoch in range(NUM_EPOCHS):
    permutation = torch.randperm(X.size(0))
    epoch_loss = 0.0
    correct = 0
    total = 0
    for i in range(num_batches):
        indices = permutation[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        batch_x = X[indices].to(device)
        batch_y = y[indices].to(device)
        
        optimizer.zero_grad()
        logits = learner(batch_x)
        loss_class = bce_loss_fn(logits, batch_y)
        
        # Orthogonality penalty: encourage C_perp to be orthogonal to W_pos_mean.
        loss_ortho = orthogonality_penalty(learner.C_perp, W_pos)
        
        loss = loss_class + LAMBDA_ORTHO * loss_ortho
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        # Accuracy (for binary classification)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)
    
    epoch_loss_avg = epoch_loss / num_batches
    accuracy = correct / total * 100
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Loss = {epoch_loss_avg:.4f}, Accuracy = {accuracy:.2f}%, Penalty = {loss_ortho.item():.4f}")

# After training, extract the learned C_perp vector.
C_perp_learned = learner.C_perp.detach().cpu().numpy()
print("Learned C_perp vector shape:", C_perp_learned.shape)
print("First 10 elements of C_perp vector:", C_perp_learned[:10])

# Optionally, save the learned vector
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, "C_perp.npy"), C_perp_learned)
print(f"Saved learned C_perp vector to {os.path.join(output_dir, 'C_perp.npy')}")

