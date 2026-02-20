"""Neural network layers.

Fully connected (Dense) layer with configurable weight initialization
and activation. Uses only Matrix and Activation.
"""

import math
import random
from nn.math_utils import Matrix
from nn.activations import Activation


class Dense:
    """A fully connected neural network layer.

    Each layer stores:
      - weights: Matrix of shape (input_size x output_size)
      - biases: Matrix of shape (1 x output_size)
      - activation: an Activation instance

    During forward pass, stores intermediate values needed for backprop:
      - self.input: the input Matrix
      - self.z: the pre-activation values (input @ weights + biases)
      - self.a: the post-activation values (activation(z))
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: Activation,
        weight_init: str = "xavier",
    ) -> None:
        """
        Initialize the layer.

        weight_init options:
          'xavier': random values scaled by sqrt(2 / (input_size + output_size))
                    Good for sigmoid and tanh.
          'he': random values scaled by sqrt(2 / input_size)
                Good for ReLU.
          'random': uniform random between -0.5 and 0.5

        Biases initialized to zeros.
        """
        if input_size <= 0 or output_size <= 0:
            raise ValueError("input_size and output_size must be positive")

        if weight_init == "xavier":
            scale = math.sqrt(2.0 / (input_size + output_size))
            self.weights = Matrix.random(
                input_size, output_size, low=-1.0, high=1.0
            ) * scale
        elif weight_init == "he":
            scale = math.sqrt(2.0 / input_size)
            self.weights = Matrix.random(
                input_size, output_size, low=-1.0, high=1.0
            ) * scale
        elif weight_init == "random":
            self.weights = Matrix.random(
                input_size, output_size, low=-0.5, high=0.5
            )
        else:
            raise ValueError(
                f"Unknown weight_init: {weight_init}. Use 'xavier', 'he', or 'random'."
            )

        # Biases: (1 x output_size)
        self.biases = Matrix([[0.0] * output_size])
        self.activation = activation

        # Filled during forward pass for backprop
        self.input: Matrix | None = None
        self.z: Matrix | None = None
        self.a: Matrix | None = None

    def forward(self, input_data: Matrix) -> Matrix:
        """
        Forward pass:
          z = input @ weights + biases (broadcast biases across all rows)
          a = activation(z)
        Store input, z, a for backprop.
        Return a.

        For bias broadcasting: biases is (1 x output_size), input is
        (batch_size x input_size), z is (batch_size x output_size).
        Add biases to each row of z.
        """
        self.input = input_data.copy()
        z = input_data.dot(self.weights)
        # Broadcast biases (1 x output_size) to (batch_size x output_size)
        batch_size = z.rows
        bias_broadcast = Matrix([self.biases.data[0][:] for _ in range(batch_size)])
        z = z + bias_broadcast
        self.z = z
        self.a = self.activation.forward(z)
        return self.a

    def backward(self, output_gradient: Matrix) -> tuple[Matrix, Matrix, Matrix]:
        """
        Backward pass: compute gradients only (no weight updates).
        1. If activation is not Softmax:
           activation_gradient = activation.derivative(self.z)
           delta = output_gradient * activation_gradient  (element-wise)
        2. If activation is Softmax:
           delta = output_gradient  (gradient already combined with loss)

        3. weights_gradient = self.input.transpose().dot(delta)
        4. biases_gradient = sum of delta across rows (sum_cols),
           giving (1 x output_size)
        5. input_gradient = delta.dot(self.weights.transpose())

        Return (weights_gradient, biases_gradient, input_gradient).
        Weight updates are applied by the optimizer.
        """
        from nn.activations import Softmax

        if isinstance(self.activation, Softmax):
            delta = output_gradient
        else:
            activation_gradient = self.activation.derivative(self.z)
            delta = output_gradient * activation_gradient

        # weights_grad: (input_size x output_size) = input.T (input_size x batch) @ delta (batch x output_size)
        weights_gradient = self.input.transpose().dot(delta)
        # biases_grad: sum over batch -> (1 x output_size)
        biases_gradient = delta.sum_cols()

        # input_gradient: (batch x input_size) = delta (batch x output_size) @ weights.T (output_size x input_size)
        input_gradient = delta.dot(self.weights.transpose())

        return (weights_gradient, biases_gradient, input_gradient)

    def get_weights(self) -> dict:
        """Return weights and biases as nested lists for serialization."""
        return {
            "weights": self.weights.to_list(),
            "biases": self.biases.to_list(),
        }

    def set_weights(self, data: dict) -> None:
        """Load weights and biases from nested lists."""
        self.weights = Matrix(data["weights"])
        self.biases = Matrix(data["biases"])

    def __repr__(self) -> str:
        return f"Dense({self.weights.rows} -> {self.weights.cols}, {self.activation})"
