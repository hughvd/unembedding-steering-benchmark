import torch
import os
from src.models.gemma_loader import load_gemma_tokenizer


activation_path = "activations"


def load_layer_acts(layer, split):

    if split not in ["train", "test"]:
        raise ValueError("Split isn't train or test!")

    inputs_list = []
    labels_list = []

    for filename in os.listdir(activation_path + f"/{split}/layer_{layer}"):

        # Load activations
        try:
            inputs = torch.load(
                os.path.join(activation_path, f"{split}/layer_{layer}", filename)
            )
        except:
            print(
                f"Failed to load: {os.path.join(activation_path, f'{split}/layer_{layer}', filename)}"
            )

        filename = filename.replace("acts", "labels")

        try:
            labels = torch.load(
                os.path.join(activation_path, f"{split}/labels", filename)
            )
        except:
            raise ValueError(
                f"Failed to load: {os.path.join(activation_path, f'{split}/layer_{layer}', filename)}"
            )

        inputs_list.append(inputs)
        labels_list.append(labels)

    try:
        output_inputs = torch.cat(inputs_list, dim=0)
        output_labels = torch.cat(labels_list, dim=0)
    except:
        output_inputs = torch.cat(inputs_list, dim=0)
        output_labels = torch.cat(
            [torch.Tensor(labels_list[i]) for i in range(len(labels_list))], dim=0
        )

    return output_inputs, output_labels


def extract_concept_mat(unembedding, words, model_name="google/gemma-2-9b"):

    # Load tokenizer
    tokenizer = load_gemma_tokenizer(model_name)

    # Set the padding side to be left in case something in the vocab is greater than one token
    tokenizer.padding_side = "left"

    # Tokenize each of the words in the dictionary and get last
    token_ids = tokenizer(
        words, return_tensors="pt", add_special_tokens=False, padding=True
    ).input_ids.squeeze()

    # Select these rows from the unembeddings
    return unembedding[token_ids]
