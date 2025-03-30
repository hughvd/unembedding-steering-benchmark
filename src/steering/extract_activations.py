from src.models.gemma_loader import load_gemma
from datasets import load_dataset
import torch

# Load the model and processor
# model, tokenizer = load_gemma(model_name="google/gemma-2-2b-it")

# Hook that stores activations of the model
activations = []


def hook(module, input, output):
    activations.append(output[0].detach())
    return output


# Register a forward hook on the each layer of the model
hooks = []
for layer in range(12):
    hooks.append(
        model.text_model.encoder.layers[layer].register_forward_hook(hook)
    )  # TODO: Change to fit Gemma architecture

# Load dataset from hugging face
sentiment_dataset = load_dataset("glue", "sst2")

# Get the train set
X_train = sentiment_dataset["train"]["sentence"]
Y_train = sentiment_dataset["train"]["label"]

# Process the train set in batches and save the activations
batch_size = 25

for i in range(0, len(X_train), batch_size):
    # Get batch of texts and labels
    batch_texts = X_train[i : i + batch_size]
    batch_labels = Y_train[i : i + batch_size]

    # Tokenize the batch
    inputs = tokenizer(
        batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    # Run the model on the batch
    outputs = model(**inputs)

# Save the activations
torch.save(activations, "train_activations.pt")

# Save the labels
torch.save(batch_labels, "train_labels.pt")

# Process the test set in batches and save the activations
for i in range(0, len(X_test), batch_size):
    # Get batch of texts and labels
    batch_texts = X_test[i : i + batch_size]
    batch_labels = Y_test[i : i + batch_size]

    # Tokenize the batch
    inputs = tokenizer(
        batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    # Run the model on the batch
    outputs = model(**inputs)

# Save the activations
torch.save(activations, "test_activations.pt")

# Save the labels
torch.save(batch_labels, "test_labels.pt")
