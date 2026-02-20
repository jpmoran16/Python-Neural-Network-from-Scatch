"""Sequential neural network implementation.

Uses Dense layers, Loss, Matrix, and one_hot_encode.
Training with full or mini-batch gradient descent.
"""

import json
import random
from nn.layers import Dense
from nn.losses import Loss
from nn.math_utils import Matrix, one_hot_encode
from nn.optimizers import Optimizer, SGD
from nn.activations import (
    Activation,
    Sigmoid,
    ReLU,
    LeakyReLU,
    Tanh,
    Softmax,
    Linear,
)


def _activation_from_name(name: str) -> Activation:
    """Build an Activation instance from class name (for load)."""
    activations = {
        "Sigmoid": lambda: Sigmoid(),
        "ReLU": lambda: ReLU(),
        "LeakyReLU": lambda: LeakyReLU(0.01),
        "Tanh": lambda: Tanh(),
        "Softmax": lambda: Softmax(),
        "Linear": lambda: Linear(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]()


class Network:
    """A sequential neural network built from Dense layers."""

    def __init__(self) -> None:
        self.layers: list[Dense] = []
        self.loss_function: Loss | None = None
        self.training_history: list[dict] = []

    def add(self, layer: Dense) -> None:
        """Add a layer to the network."""
        self.layers.append(layer)

    def set_loss(self, loss: Loss) -> None:
        """Set the loss function."""
        self.loss_function = loss

    def forward(self, X: Matrix) -> Matrix:
        """Forward pass through all layers sequentially."""
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(
        self,
        loss_gradient: Matrix,
        learning_rate: float,
        optimizer: Optimizer | None = None,
    ) -> None:
        """Backward pass through all layers in reverse order.
        If optimizer is provided, use it for weight updates; otherwise
        use vanilla SGD with learning_rate."""
        grad = loss_gradient
        opt = optimizer if optimizer is not None else SGD(learning_rate=learning_rate)
        for layer in reversed(self.layers):
            weight_grad, bias_grad, input_grad = layer.backward(grad)
            opt.step(layer, weight_grad, bias_grad)
            grad = input_grad

    def train(
        self,
        X: Matrix,
        y: Matrix,
        epochs: int,
        learning_rate: float,
        batch_size: int | None = None,
        optimizer: Optimizer | None = None,
        verbose: bool = True,
        print_every: int = 100,
    ) -> list[dict]:
        """
        Train the network.

        If batch_size is None, use full batch (all data at once).
        If batch_size is set, use mini-batch gradient descent:
          - Shuffle data each epoch
          - Split into batches of batch_size
          - Forward + backward on each batch

        If optimizer is provided, it is used for weight updates; otherwise
        vanilla SGD with learning_rate is used.

        Each epoch:
          1. Forward pass
          2. Compute loss
          3. Backward pass (loss gradient -> layer gradients -> optimizer step)
          4. Store epoch number, loss value in training_history
          5. If verbose and epoch % print_every == 0:
             print(f"Epoch {epoch}/{epochs} — Loss: {loss:.6f}")

        Return training_history.
        """
        if self.loss_function is None:
            raise ValueError("Set loss function with set_loss() before training")

        self.training_history = []
        n_samples = X.rows

        for epoch in range(1, epochs + 1):
            if batch_size is None:
                X_batches, y_batches = [X], [y]
            else:
                X_shuf, y_shuf = self._shuffle_data(X, y)
                X_batches, y_batches = self._create_batches(X_shuf, y_shuf, batch_size)

            epoch_loss = 0.0
            num_batches = len(X_batches)

            for X_batch, y_batch in zip(X_batches, y_batches):
                predictions = self.forward(X_batch)
                loss_val = self.loss_function.forward(predictions, y_batch)
                epoch_loss += loss_val
                loss_grad = self.loss_function.backward(predictions, y_batch)
                self.backward(loss_grad, learning_rate, optimizer)

            avg_loss = epoch_loss / num_batches
            self.training_history.append({"epoch": epoch, "loss": avg_loss})

            if verbose and (epoch % print_every == 0 or epoch == 1 or epoch == epochs):
                print(f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.6f}")

        return self.training_history

    def predict(self, X: Matrix) -> Matrix:
        """Forward pass without training. Returns output matrix."""
        return self.forward(X)

    def evaluate(self, X: Matrix, y: Matrix) -> dict:
        """
        Evaluate the network on test data.
        Returns dict with:
          - 'loss': the loss value
          - 'accuracy': classification accuracy (if applicable)
            Compare argmax of predictions vs argmax of targets per row
        """
        predictions = self.forward(X)
        loss_val = self.loss_function.forward(predictions, y) if self.loss_function else 0.0

        pred_indices = predictions.max_index_per_row()
        target_indices = y.max_index_per_row()
        correct = sum(1 for p, t in zip(pred_indices, target_indices) if p == t)
        accuracy = correct / X.rows if X.rows else 0.0

        return {"loss": loss_val, "accuracy": accuracy}

    def summary(self) -> None:
        """Print a formatted summary of the network architecture.
        Show each layer, input/output sizes, activation, and param count.
        Show total parameters at the end."""
        total_params = 0
        print("Network summary")
        print("-" * 50)
        for i, layer in enumerate(self.layers):
            w = layer.weights
            b = layer.biases
            params = w.rows * w.cols + b.rows * b.cols
            total_params += params
            print(f"Layer {i + 1}: Dense({w.rows} -> {w.cols}, {layer.activation})  params={params}")
        print("-" * 50)
        print(f"Total parameters: {total_params}")

    def save(self, filepath: str) -> None:
        """Save model architecture and weights to a JSON file."""
        layers_data = []
        for layer in self.layers:
            act = layer.activation
            act_name = act.__class__.__name__
            entry = {
                "input_size": layer.weights.rows,
                "output_size": layer.weights.cols,
                "activation": act_name,
                "weights": layer.weights.to_list(),
                "biases": layer.biases.to_list(),
            }
            if act_name == "LeakyReLU":
                entry["activation_kwargs"] = {"alpha": getattr(act, "alpha", 0.01)}
            layers_data.append(entry)
        data = {
            "layers": layers_data,
            "loss": self.loss_function.__class__.__name__ if self.loss_function else None,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        """Load model architecture and weights from a JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        self.layers = []
        for layer_cfg in data["layers"]:
            in_size = layer_cfg["input_size"]
            out_size = layer_cfg["output_size"]
            act_name = layer_cfg["activation"]
            kwargs = layer_cfg.get("activation_kwargs", {})
            if act_name == "LeakyReLU":
                act = LeakyReLU(alpha=kwargs.get("alpha", 0.01))
            else:
                act = _activation_from_name(act_name)
            layer = Dense(in_size, out_size, act, weight_init="random")
            layer.set_weights({"weights": layer_cfg["weights"], "biases": layer_cfg["biases"]})
            self.layers.append(layer)
        self.training_history = []
        # Loss is not restored (caller can set_loss again)

    def _shuffle_data(self, X: Matrix, y: Matrix) -> tuple[Matrix, Matrix]:
        """Shuffle X and y rows together (same permutation)."""
        n = X.rows
        indices = list(range(n))
        random.shuffle(indices)
        X_data = [X.data[i][:] for i in indices]
        y_data = [y.data[i][:] for i in indices]
        return Matrix(X_data), Matrix(y_data)

    def _create_batches(
        self, X: Matrix, y: Matrix, batch_size: int
    ) -> tuple[list[Matrix], list[Matrix]]:
        """Split X and y into list of (X_batch, y_batch) tuples. Returns (X_batches, y_batches)."""
        X_batches = []
        y_batches = []
        for start in range(0, X.rows, batch_size):
            end = min(start + batch_size, X.rows)
            X_batch = Matrix([X.data[i][:] for i in range(start, end)])
            y_batch = Matrix([y.data[i][:] for i in range(start, end)])
            X_batches.append(X_batch)
            y_batches.append(y_batch)
        return X_batches, y_batches
