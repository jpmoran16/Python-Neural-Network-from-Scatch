"""Activation functions for neural networks.

Uses only Matrix from math_utils and the math standard library.
Each activation provides forward pass and derivative for backpropagation.
"""

import math
from nn.math_utils import Matrix


class Activation:
    """Base class for activation functions."""

    def forward(self, z: Matrix) -> Matrix:
        """Compute activation output. Subclasses must override."""
        raise NotImplementedError

    def derivative(self, z: Matrix) -> Matrix:
        """Compute derivative w.r.t. input z for backprop. Subclasses must override."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__


class Sigmoid(Activation):
    """Sigmoid: f(x) = 1 / (1 + e^(-x))
    Derivative: f'(x) = f(x) * (1 - f(x))
    Clip input to range [-500, 500] to prevent math overflow."""

    def forward(self, z: Matrix) -> Matrix:
        z_clipped = z.clip(-500.0, 500.0)
        return z_clipped.apply(lambda x: 1.0 / (1.0 + math.exp(-x)))

    def derivative(self, z: Matrix) -> Matrix:
        z_clipped = z.clip(-500.0, 500.0)
        s = z_clipped.apply(lambda x: 1.0 / (1.0 + math.exp(-x)))
        rows, cols = s.shape()
        ones = Matrix([[1.0] * cols for _ in range(rows)])
        return s * (ones - s)


class ReLU(Activation):
    """ReLU: f(x) = max(0, x)
    Derivative: f'(x) = 1 if x > 0, else 0"""

    def forward(self, z: Matrix) -> Matrix:
        return z.apply(lambda x: max(0.0, x))

    def derivative(self, z: Matrix) -> Matrix:
        return z.apply(lambda x: 1.0 if x > 0 else 0.0)


class LeakyReLU(Activation):
    """LeakyReLU: f(x) = x if x > 0, else alpha * x
    Default alpha = 0.01
    Derivative: f'(x) = 1 if x > 0, else alpha"""

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def forward(self, z: Matrix) -> Matrix:
        return z.apply(lambda x: x if x > 0 else self.alpha * x)

    def derivative(self, z: Matrix) -> Matrix:
        return z.apply(lambda x: 1.0 if x > 0 else self.alpha)


class Tanh(Activation):
    """Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    Derivative: f'(x) = 1 - f(x)^2
    Clip input to range [-500, 500] to prevent overflow."""

    def forward(self, z: Matrix) -> Matrix:
        z_clipped = z.clip(-500.0, 500.0)
        return z_clipped.apply(math.tanh)

    def derivative(self, z: Matrix) -> Matrix:
        z_clipped = z.clip(-500.0, 500.0)
        f = z_clipped.apply(math.tanh)
        rows, cols = f.shape()
        ones = Matrix([[1.0] * cols for _ in range(rows)])
        return ones - (f * f)


class Softmax(Activation):
    """Softmax: f(x_i) = e^(x_i - max(x)) / sum(e^(x_j - max(x)))
    Applied per ROW (each row is one sample).
    Subtract max per row for numerical stability BEFORE exponentiating.
    Derivative is handled differently in backprop (combined with
    cross-entropy loss), so derivative() returns a matrix of ones;
    in practice the gradient is computed as (output - target) when
    using softmax + cross-entropy together."""

    def forward(self, z: Matrix) -> Matrix:
        z_clipped = z.clip(-500.0, 500.0)
        result_data = []
        for i in range(z_clipped.rows):
            row = [z_clipped.data[i][j] for j in range(z_clipped.cols)]
            max_r = max(row)
            row_shifted = [v - max_r for v in row]
            row_exp = [math.exp(v) for v in row_shifted]
            s = sum(row_exp)
            s = s if s > 0 else 1e-10
            result_data.append([v / s for v in row_exp])
        return Matrix(result_data)

    def derivative(self, z: Matrix) -> Matrix:
        """Returns ones of same shape. Use (output - target) as gradient when
        softmax is used with cross-entropy loss in backprop."""
        rows, cols = z.shape()
        return Matrix([[1.0] * cols for _ in range(rows)])


class Linear(Activation):
    """Linear (identity): f(x) = x
    Derivative: f'(x) = 1
    Used for regression output layers."""

    def forward(self, z: Matrix) -> Matrix:
        return z.copy()

    def derivative(self, z: Matrix) -> Matrix:
        rows, cols = z.shape()
        return Matrix([[1.0] * cols for _ in range(rows)])
