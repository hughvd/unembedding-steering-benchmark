from src.models.gemma_loader import load_gemma
import torch
from tqdm import tqdm
import os

# 1) Adjective & verb pools
positive_adjectives = [
    " perfect",
    " fantastic",
    " delightful",
    " cheerful",
    " good",
    " remarkable",
    " satisfactory",
    " wonderful",
    " nice",
    " fabulous",
    " outstanding",
    " satisfying",
    " awesome",
    " exceptional",
    " adequate",
    " incredible",
    " extraordinary",
    " amazing",
    " decent",
    " lovely",
    " brilliant",
    " charming",
    " terrific",
    " superb",
    " spectacular",
    " great",
    " splendid",
    " beautiful",
    " positive",
    " excellent",
    " pleasant",
]
negative_adjectives = [
    " dreadful",
    " bad",
    " dull",
    " depressing",
    " miserable",
    " tragic",
    " nasty",
    " inferior",
    " horrific",
    " terrible",
    " ugly",
    " disgusting",
    " disastrous",
    " annoying",
    " boring",
    " offensive",
    " frustrating",
]
positive_verbs = [" enjoyed", " loved", " liked", " appreciated", " admired"]
negative_verbs = [" hated", " disliked", " despised"]

# 2) Build dataset with adj and verb tags
dataset = []
for adj in positive_adjectives:
    for verb in positive_verbs:
        prompt = (
            f"I thought this movie was{adj}, I{verb} it.\nConclusion: This movie is"
        )
        dataset.append((prompt, 1, adj, verb))
for adj in negative_adjectives:
    for verb in negative_verbs:
        prompt = (
            f"I thought this movie was{adj}, I{verb} it.\nConclusion: This movie is"
        )
        dataset.append((prompt, 0, adj, verb))

prompts, labels, adjs, verbs = zip(*dataset)
prompts = list(prompts)
labels = list(labels)
adjs = list(adjs)
verbs = list(verbs)

print("Total examples:", len(prompts))  # should be 206

# 3) Prepare directories
model, tokenizer = load_gemma(model_name="google/gemma-2-2b")
model = model.to("cuda")
num_layers = len(model.model.layers)
base_dir = "activations/toy_mov"
for layer_idx in range(num_layers):
    os.makedirs(f"{base_dir}/adj_layer_{layer_idx}", exist_ok=True)
    os.makedirs(f"{base_dir}/verb_layer_{layer_idx}", exist_ok=True)
os.makedirs(f"{base_dir}/labels", exist_ok=True)

# 4) Hook for residual stream
activations = []


def hook(module, inp, out):
    activations.append(out[0].detach().cpu())
    return out


hooks = [layer.register_forward_hook(hook) for layer in model.model.layers]

# Define baseline adj position and verb position
def_adj_pos = 6
def_verb_pos = 9

# 5) Batch process
batch_size = 16
for start in tqdm(range(0, len(prompts), batch_size), desc="Extract activations"):
    end = start + batch_size
    batch_prompts = prompts[start:end]
    batch_labels = labels[start:end]
    batch_adjs = adjs[start:end]
    batch_verbs = verbs[start:end]

    # tokenize
    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to("cuda")

    # forward
    with torch.no_grad():
        _ = model(**inputs)

    # move ids to CPU
    input_ids = inputs["input_ids"].cpu()

    # find token positions
    adj_positions = [
        def_adj_pos + len(tokenizer.encode(a, add_special_tokens=False)) - 1
        for a in batch_adjs
    ]
    verb_positions = [
        def_verb_pos + len(tokenizer.encode(v, add_special_tokens=False)) - 1
        for v in batch_verbs
    ]

    # extract & save per layer
    for layer_idx, layer_act in enumerate(activations):
        # stack per-batch activations
        adj_acts = torch.stack(
            [layer_act[i, pos] for i, pos in enumerate(adj_positions)]
        )
        verb_acts = torch.stack(
            [layer_act[i, pos] for i, pos in enumerate(verb_positions)]
        )
        torch.save(adj_acts, f"{base_dir}/adj_layer_{layer_idx}/batch_{start}.pt")
        torch.save(verb_acts, f"{base_dir}/verb_layer_{layer_idx}/batch_{start}.pt")

    # save labels
    torch.save(torch.tensor(batch_labels), f"{base_dir}/labels/batch_{start}.pt")

    # clear
    activations.clear()
    torch.cuda.empty_cache()

# remove hooks
for h in hooks:
    h.remove()

print("Done extracting adjective (train) and verb (test) activations.")
