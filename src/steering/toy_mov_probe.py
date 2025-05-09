import os
import glob
import torch
import wandb
from src.steering.probe import LinearProbe, LinearProbePerp
from src.steering.probe_utils import extract_concept_mat

BASE_DIR = "activations/toy_mov"


def load_activations(layer: int, mode: str):
    # mode = "train" uses adjective activations; mode = "test" uses verb activations
    folder = f"adj_layer_{layer}" if mode == "train" else f"verb_layer_{layer}"
    act_dir = os.path.join(BASE_DIR, folder)
    files = sorted(
        glob.glob(os.path.join(act_dir, "batch_*.pt")),
        key=lambda fn: int(os.path.basename(fn).split("_")[1].split(".")[0]),
    )
    X = torch.cat([torch.load(f) for f in files], dim=0)
    # labels are the same for train/test
    label_dir = os.path.join(BASE_DIR, "labels")
    labs = sorted(
        glob.glob(os.path.join(label_dir, "batch_*.pt")),
        key=lambda fn: int(os.path.basename(fn).split("_")[1].split(".")[0]),
    )
    y = torch.cat([torch.load(f) for f in labs], dim=0).long()
    return X, y


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = {
        "input_dim": 2304,  # e.g., 2048
        "output_dim": 2,
        "epochs": 50,
        "learning_rate": 1e-3,
        "perp": False,
    }

    # determine number of layers
    layer_dirs = [d for d in os.listdir(BASE_DIR) if d.startswith("adj_layer_")]
    num_layers = max(int(d.split("_")[2]) for d in layer_dirs) + 1

    for layer in range(num_layers):
        X_train, y_train = load_activations(layer, "train")
        X_test, y_test = load_activations(layer, "test")

        if cfg["perp"]:
            # Use perpendicular probe

            # Get the c subspace
            unembedding = torch.load("src/steering/gemma-2-2b-unembed.pt")
            pos_subspace = extract_concept_mat(
                unembedding,
                [
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
                ],
            )

            neg_subspace = extract_concept_mat(
                unembedding,
                [
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
                    " wretched",
                    " inadequate",
                    " dire",
                    " unpleasant",
                    " horrible",
                    " disappointing",
                    " awful",
                ],
            )
            probe = LinearProbePerp(
                cfg["input_dim"], cfg["output_dim"], neg_subspace, pos_subspace
            )
            wandb.init(
                project="perp_toy_mov_probe", name=f"perp_layer_{layer}", reinit=True
            )
        else:
            # Use standard probe
            probe = LinearProbe(cfg["input_dim"], cfg["output_dim"])
            wandb.init(project="toy_mov_probe", name=f"layer_{layer}", reinit=True)

        probe.fit(
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=cfg["epochs"],
            learning_rate=cfg["learning_rate"],
        )

        if cfg["perp"]:
            torch.save(
                probe.state_dict(),
                f"results/toy_mov_probes/perp_toy_mov_probe_layer_{layer}.pth",
            )
        else:
            torch.save(
                probe.state_dict(),
                f"results/toy_mov_probes/toy_mov_probe_layer_{layer}.pth",
            )
        wandb.finish()
