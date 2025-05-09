import os
import glob
import torch
import numpy as np

# Paths
BASE_DIR = "activations/toy_mov"
OUT_DIR = "results/toy_mov_caa"
os.makedirs(OUT_DIR, exist_ok=True)


def load_activations(layer: int, mode: str):
    """
    Load activations for a given layer and mode.
    mode = "train" loads adjective activations, mode = "test" loads verb activations.
    Returns X (NxD) and labels y (N,).
    """
    # choose folder
    folder = f"adj_layer_{layer}" if mode == "train" else f"verb_layer_{layer}"
    act_dir = os.path.join(BASE_DIR, folder)
    files = sorted(
        glob.glob(os.path.join(act_dir, "batch_*.pt")),
        key=lambda fn: int(os.path.basename(fn).split("_")[1].split(".")[0]),
    )
    X = torch.cat([torch.load(f) for f in files], dim=0)
    # labels are shared
    label_dir = os.path.join(BASE_DIR, "labels")
    labs = sorted(
        glob.glob(os.path.join(label_dir, "batch_*.pt")),
        key=lambda fn: int(os.path.basename(fn).split("_")[1].split(".")[0]),
    )
    y = torch.cat([torch.load(f) for f in labs], dim=0).long()
    return X, y


if __name__ == "__main__":
    # detect number of layers
    layer_dirs = [d for d in os.listdir(BASE_DIR) if d.startswith("adj_layer_")]
    num_layers = max(int(d.split("_")[2]) for d in layer_dirs) + 1

    for layer in range(num_layers):
        X_adj, y = load_activations(layer, "train")
        X_verb, _ = load_activations(layer, "test")

        # Compute class-based means & stds
        for name, X in [("adj", X_adj), ("verb", X_verb)]:
            pos = X[y == 1]
            neg = X[y == 0]

            mean_pos = pos.mean(dim=0)
            mean_neg = neg.mean(dim=0)
            std_pos = pos.std(dim=0)
            std_neg = neg.std(dim=0)
            diff = mean_pos - mean_neg

            # save
            # torch.save(
            #     mean_pos, os.path.join(OUT_DIR, f"mean_pos_{name}_layer_{layer}.pt")
            # )
            # torch.save(
            #     mean_neg, os.path.join(OUT_DIR, f"mean_neg_{name}_layer_{layer}.pt")
            # )
            # torch.save(
            #     std_pos, os.path.join(OUT_DIR, f"std_pos_{name}_layer_{layer}.pt")
            # )
            # torch.save(
            #     std_neg, os.path.join(OUT_DIR, f"std_neg_{name}_layer_{layer}.pt")
            # )
            torch.save(diff, os.path.join(OUT_DIR, f"diff_{name}_layer_{layer}.pt"))

        print(f"Layer {layer}: saved CAA vectors for adj & verb.")
