# !pip install mlflow torch datasets transformers scikit-learn numpy wandb

import numpy as np
import torch
import random
import mlflow
import mlflow.pytorch
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import wandb
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

mlflow.set_experiment("Activation Steering Experiment with Gemma")
run = mlflow.start_run()
mlflow.log_param("model", "google/gemma-2b")
mlflow.log_param("dataset", "SST-2 (GLUE)")
mlflow.log_param("concept", "positive sentiment")

wandb.init(project="activation_steering_experiment", config={
    "model": "google/gemma-2b",
    "dataset": "SST-2 (GLUE)",
    "experiment": "Activation Steering with Gemma SAEs and 50-token rollout"
})

# load gemma and its tokenizer
model_name = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
model.eval()

# this contains the public SAEs trained on the Gemma-2B residual stream.
sae_model_name = "google/gemma-scope-2b-pt-res"
sae_model = AutoModel.from_pretrained(sae_model_name)
# assumes that the SAE model exposes a dictionary "public_saes" w/ keys foreach concept direction
gemma_public_sae = sae_model.public_saes["positive"]
mlflow.log_param("gemma_public_sae_shape", str(gemma_public_sae.shape))
print("Loaded Gemma public SAE for positive sentiment with shape:", gemma_public_sae.shape)

# load dataset SST2
dataset = load_dataset("glue", "sst2")
n_samples = 200
train_sentences = dataset['train']['sentence'][:n_samples]
train_labels = dataset['train']['label'][:n_samples]
print(f"Using {len(train_sentences)} samples for the experiment.")
mlflow.log_metric("num_samples", len(train_sentences))

# gemma hidden state extractions
def extract_hidden_state(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    #avg the final layer's hidden states over the token sequence
    hidden = outputs.hidden_states[-1].mean(dim=1).squeeze().detach().numpy()
    return hidden

hidden_states = []
for text in train_sentences:
    try:
        h = extract_hidden_state(text)
        hidden_states.append(h)
    except Exception as e:
        print(f"Error extracting hidden state for text: {text} - {e}")
hidden_states = np.array(hidden_states)
labels = np.array(train_labels)
print("Extracted hidden states shape:", hidden_states.shape)

# train probe

clf = LogisticRegression(max_iter=1000)
clf.fit(hidden_states, labels)
C = clf.coef_.flatten()  # Learned concept (steering) vector
print("Trained linear probe. Learned concept vector C shape:", C.shape)
mlflow.log_metric("C_norm", np.linalg.norm(C))

#extract positive sentiment vectors
positive_tokens = ["positive", "good", "great", "amazing", "excellent"]
W_pos_vectors = []
for token in positive_tokens:
    token_id = tokenizer.encode(token)[0]
    vec = model.transformer.wte.weight[token_id].detach().numpy()
    W_pos_vectors.append(vec)
W_pos_vectors = np.stack(W_pos_vectors)
print("Collected positive token unembedding vectors shape:", W_pos_vectors.shape)
mlflow.log_param("positive_tokens", positive_tokens)

#vector aggregation
W_pos_mean = np.mean(W_pos_vectors, axis=0)
pca = PCA(n_components=1)
pca.fit(W_pos_vectors)
W_pos_pc1 = pca.components_[0]

# compare steering dirs
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim_mean = cosine_similarity(C, W_pos_mean)
sim_pc1 = cosine_similarity(C, W_pos_pc1)
print("Cosine similarity between C and W_pos_mean:", sim_mean)
print("Cosine similarity between C and W_pos_pc1:", sim_pc1)
mlflow.log_metric("cosine_similarity_mean", sim_mean)
mlflow.log_metric("cosine_similarity_pc1", sim_pc1)

sim_public = cosine_similarity(C, gemma_public_sae)
print("Cosine similarity between learned C and Gemma public SAE (positive):", sim_public)
mlflow.log_metric("cosine_similarity_public_SAE", sim_public)

# CAA and act steer intervention
def intervene_and_generate(text, steering_vector, alpha=1.0):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1].mean(dim=1).squeeze().detach()
    hidden_modified = hidden + alpha * torch.tensor(steering_vector, dtype=hidden.dtype)
    logits = hidden_modified @ model.transformer.wte.weight.T
    next_token_id = torch.argmax(logits).item()
    next_token = tokenizer.decode([next_token_id])
    return next_token

sample_text = "The movie was"
gen_C = intervene_and_generate(sample_text, C, alpha=1.0)
gen_Wpos = intervene_and_generate(sample_text, W_pos_mean, alpha=1.0)
print("Generated token with steering C:", gen_C)
print("Generated token with steering W_pos_mean:", gen_Wpos)
mlflow.log_param("generated_token_C", gen_C)
mlflow.log_param("generated_token_Wpos", gen_Wpos)

def compute_caa(text, concept_vector, alpha=1.0):
    inputs = tokenizer(text, return_tensors="pt")
    baseline_out = model(**inputs, output_hidden_states=True)
    baseline_hidden = baseline_out.hidden_states[-1].mean(dim=1).squeeze().detach()
    baseline_logits = baseline_hidden @ model.transformer.wte.weight.T
    baseline_probs = torch.softmax(baseline_logits, dim=-1)
    
    intervened_hidden = baseline_hidden + alpha * torch.tensor(concept_vector, dtype=baseline_hidden.dtype)
    intervened_logits = intervened_hidden @ model.transformer.wte.weight.T
    intervened_probs = torch.softmax(intervened_logits, dim=-1)
    
    differences = {}
    for token in positive_tokens:
        token_id = tokenizer.encode(token)[0]
        diff = intervened_probs[token_id].item() - baseline_probs[token_id].item()
        differences[token] = diff
    return differences

caa_diffs = compute_caa(sample_text, C, alpha=1.0)
print("CAA differences for sample text:", caa_diffs)
mlflow.log_metric("caa_diff_positive", np.mean(list(caa_diffs.values())))

# WandB plot for the CAA diffs
fig, ax = plt.subplots()
tokens = list(caa_diffs.keys())
diffs = [caa_diffs[t] for t in tokens]
ax.bar(tokens, diffs)
ax.set_ylabel("Probability Difference")
ax.set_title("CAA: Change in Token Probabilities After Intervention")
plt.tight_layout()
wandb.log({"CAA_Probability_Differences": wandb.Image(fig)})
plt.show()

# 50 token rollout -- this is our MDP
def generate_rollout(prompt, steering_vector=None, alpha=1.0, length=50):
    generated_tokens = []
    current_prompt = prompt
    for i in range(length):
        inputs = tokenizer(current_prompt, return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1].mean(dim=1).squeeze().detach()
        if steering_vector is not None:
            hidden = hidden + alpha * torch.tensor(steering_vector, dtype=hidden.dtype)
        logits = hidden @ model.transformer.wte.weight.T
        next_token_id = torch.argmax(logits).item()
        next_token = tokenizer.decode([next_token_id])
        generated_tokens.append(next_token)
        current_prompt += next_token
    return generated_tokens

baseline_rollout = generate_rollout("The movie was", steering_vector=None, length=50)
intervened_rollout = generate_rollout("The movie was", steering_vector=C, alpha=1.0, length=50)

print("Baseline Rollout:\n", "".join(baseline_rollout))
print("\nIntervened Rollout (with C):\n", "".join(intervened_rollout))

wandb.log({
    "Baseline_Rollout": "".join(baseline_rollout),
    "Intervened_Rollout": "".join(intervened_rollout)
})

mlflow.end_run()
wandb.finish()
print("Experiment run logged with MLflow and WandB.")
