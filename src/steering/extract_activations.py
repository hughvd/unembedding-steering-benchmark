from src.models.gemma_loader import load_gemma
from datasets import load_dataset
import torch
from tqdm import tqdm

# Load the model and processor
model, tokenizer = load_gemma(model_name="google/gemma-2-2b")

model.to("cuda")
model.eval()

# Hook that stores activations of the model
activations = []


def hook(module, input, output):
    activations.append(output[0].detach().cpu())
    return output


# Register a forward hook on the each layer of the model
hooks = []

for layer in range(len(model.model.layers)):
    hooks.append(model.model.layers[layer].register_forward_hook(hook))

# Load dataset from hugging face
print("Loading dataset")
sentiment_dataset = load_dataset("glue", "sst2")

print("Dataset loaded")

# Get the train set
X_train = sentiment_dataset["train"]["sentence"]
Y_train = sentiment_dataset["train"]["label"]

# Get the test set
X_test = sentiment_dataset["validation"]["sentence"]
Y_test = sentiment_dataset["validation"]["label"]

print("Train and test set loaded")

# Process the train set in batches and save the activations
batch_size = 10
max_samples = 50000

for i in tqdm(
    range(0, min(max_samples, len(X_train)), batch_size),
    desc="Processing train set",
):
    # Get batch of texts and labels
    batch_texts = X_train[i : i + batch_size]
    batch_labels = Y_train[i : i + batch_size]

    # Tokenize the batch
    inputs = tokenizer(
        batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to("cuda")

    # Run the model on the batch
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the activations from the last non padded token
    last_non_padded_indices = (inputs["attention_mask"].sum(dim=1) - 1).tolist()
    last_activations = torch.stack(
        [
            torch.stack(
                [
                    activations[layer][batch_idx, idx]
                    for batch_idx, idx in enumerate(last_non_padded_indices)
                ]
            )
            for layer in range(len(activations))
        ]
    )

    # Save the activations
    for layer, acts in enumerate(last_activations):
        torch.save(acts, f"activations/train/layer_{layer}/acts_{i}.pt")

    # Save the labels
    torch.save(torch.Tensor(batch_labels), f"activations/train/labels/labels_{i}.pt")

    activations.clear()
    torch.cuda.empty_cache()

# Process the test set in batches and save the activations
for i in tqdm(
    range(0, min(max_samples, len(X_test)), batch_size),
    desc="Processing test set",
):
    # Get batch of texts and labels
    batch_texts = X_test[i : i + batch_size]
    batch_labels = Y_test[i : i + batch_size]

    # Tokenize the batch
    inputs = tokenizer(
        batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to("cuda")

    # Run the model on the batch
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the activations from the last non padded token
    last_non_padded_indices = (inputs["attention_mask"].sum(dim=1) - 1).tolist()
    last_activations = torch.stack(
        [
            torch.stack(
                [
                    activations[layer][batch_idx, idx]
                    for batch_idx, idx in enumerate(last_non_padded_indices)
                ]
            )
            for layer in range(len(activations))
        ]
    )

    # Save the activations
    for layer, acts in enumerate(last_activations):
        torch.save(acts, f"activations/test/layer_{layer}/acts_{i}.pt")

    # Save the labels
    torch.save(torch.Tensor(batch_labels), f"activations/test/labels/labels_{i}.pt")

    activations.clear()
    torch.cuda.empty_cache()
