import torch
import wandb
from src.steering.probe_utils import load_layer_acts, extract_concept_mat


# Probe Class
class LinearProbe(torch.nn.Module):
    """
    A linear probe for classification tasks.
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize the linear probe.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Number of classes for classification.
        """

        super(LinearProbe, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)

    def loss_fn(self, outputs, labels):
        loss = torch.nn.CrossEntropyLoss()
        return loss(outputs, labels)

    def forward(self, x):
        """
        Forward pass through the linear probe.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Output probabilities.
        """
        x = self.linear(x)
        return x

    def evaluate(self, X_test, y_test, epoch):
        """
        Compute both loss and accuracy on the test set.
        """
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        loader = torch.utils.data.DataLoader(list(zip(X_test, y_test)), batch_size=32)
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self(X_batch)
                loss = self.loss_fn(outputs, y_batch)
                total_loss += loss.item() * y_batch.size(0)

                preds = outputs.argmax(dim=1)
                total_correct += (preds == y_batch).sum().item()
                total_samples += y_batch.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        wandb.log({"epoch": epoch, "eval_loss": avg_loss, "eval_accuracy": accuracy})
        print(f"Epoch {epoch:2d} — eval loss: {avg_loss:.4f}, eval acc: {accuracy:.4f}")
        return avg_loss, accuracy

    def fit(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs: int = 10,
        learning_rate: float = 0.001,
    ):
        """
        Fit the linear probe to the data.

        Args:
            activation_path (str): Path to the directory containing activation files.
            layer (int): Layer index to use for the probe.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            num_batches = 0

            # Iterate through folder with activations
            for X_batch, y_batch in torch.utils.data.DataLoader(
                list(zip(X_train, y_train)), batch_size=32, shuffle=True
            ):
                # Forward pass
                outputs = self(X_batch)

                loss = self.loss_fn(outputs, y_batch.long())

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            # Log metrics to wandb
            avg_loss = epoch_loss / num_batches
            wandb.log(
                {"epoch": epoch, "train loss": avg_loss, "learning_rate": learning_rate}
            )

            # Measure test set acchracuy
            test_loss = self.evaluate(X_test, y_test, epoch)

            print(
                "Epoch:",
                epoch,
                "Train Loss:",
                avg_loss,
                "Test Loss:",
                test_loss,
            )


class LinearProbePerp(LinearProbe):
    """
    A linear probe for classification tasks.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        unembed_vocab_0: torch.Tensor,
        unembed_vocab_1: torch.Tensor,
        alpha: float = 1.0,
    ):
        super().__init__(input_dim, output_dim)
        # assume unembed_vocab is [V, D]; we'll penalize W @ U^T
        self.register_buffer("U_0", unembed_vocab_0)
        self.register_buffer("U_1", unembed_vocab_1)
        self.alpha = alpha

    def loss_fn(self, outputs, labels):
        # Base cross‐entropy
        base = super().loss_fn(outputs, labels)
        # Dot product for positive class
        dot_0 = self.U_0 @ self.linear.weight[0]
        # Dot product for negative class'
        dot_1 = self.U_1 @ self.linear.weight[1]
        # Sum of both losses
        penalty = dot_0.norm(1) + dot_1.norm(1)
        # Weighted full loss
        return base + self.alpha * penalty


cfg = {
    "input_dim": 2304,
    "output_dim": 2,
    "epochs": 100,
    "learning_rate": 0.0001,
    "activation_path": "activations/",
}


# # Gets a probe for each layer
# if __name__ == "__main__":

#     # Example usage
#     input_dim = cfg["input_dim"]  # Example input dimension
#     output_dim = cfg["output_dim"]

#     for i in range(12, 26):

#         # Initialize Probe
#         probe = LinearProbe(input_dim, output_dim)

#         # Load data
#         X_train, y_train = load_layer_acts(i, "train")
#         X_test, y_test = load_layer_acts(i, "test")

#         wandb.init(
#             project="sentiment_probe",
#             name=f"layer_{i}",
#             reinit=True,
#         )

#         # Fit the probe
#         probe.fit(
#             X_train,
#             y_train,
#             X_test,
#             y_test,
#             epochs=cfg["epochs"],
#             learning_rate=cfg["learning_rate"],
#         )

#         # Save the model
#         torch.save(probe.state_dict(), f"sentiment_linear_probe_{i}.pth")


# Gets a c_perp probe for each layer
if __name__ == "__main__":

    # Example usage
    input_dim = cfg["input_dim"]  # Example input dimension
    output_dim = cfg["output_dim"]

    # Get the c subspace
    unembedding = torch.load("src/steering/gemma-2-2b-unembed.pt")
    pos_subspace = extract_concept_mat(
        unembedding,
        [
            " happy",
            " amazing",
            " splendid",
            " incredible",
            " joyful",
        ],
    )

    neg_subspace = extract_concept_mat(
        unembedding,
        [
            " sad",
            " horrible",
            " bad",
            " terrible",
            " worst",
        ],
    )

    for i in range(12, 26):

        # Initialize Probe
        probe = LinearProbePerp(input_dim, output_dim, neg_subspace, pos_subspace)

        # Load data
        X_train, y_train = load_layer_acts(i, "train")
        X_test, y_test = load_layer_acts(i, "test")

        wandb.init(
            project="sentiment_probe",
            name=f"perp_layer_{i}",
            reinit=True,
        )

        # Fit the probe
        probe.fit(
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=cfg["epochs"],
            learning_rate=cfg["learning_rate"],
        )

        # Save the model
        torch.save(probe.state_dict(), f"perp_sentiment_linear_probe_{i}.pth")
