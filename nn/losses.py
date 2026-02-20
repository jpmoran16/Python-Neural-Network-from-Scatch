"""Loss functions for neural networks.

Uses only Matrix from math_utils and the math standard library.
Each loss provides forward (scalar loss value) and backward (gradient w.r.t. predictions).
"""

import math
from nn.math_utils import Matrix


# Clip bounds for log stability (avoid log(0))
_EPS = 1e-15
_EPS_HIGH = 1.0 - _EPS


class Loss:
    """Base class for loss functions."""

    def forward(self, predictions: Matrix, targets: Matrix) -> float:
        """Compute loss value. Subclasses must override."""
        raise NotImplementedError

    def backward(self, predictions: Matrix, targets: Matrix) -> Matrix:
        """Compute gradient dL/d(predictions). Subclasses must override."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__


def _check_shape(predictions: Matrix, targets: Matrix) -> None:
    """Raise ValueError if predictions and targets have different shapes."""
    if predictions.shape() != targets.shape():
        raise ValueError(
            f"predictions shape {predictions.shape()} does not match targets shape {targets.shape()}"
        )


class MSE(Loss):
    """Mean Squared Error for regression.

    Forward: L = (1/n) * sum((predictions - targets)^2)
    Backward: dL/dpred = (2/n) * (predictions - targets)

    Where n = total number of elements.
    Gradient derivation: d/dpred (pred - target)^2 = 2(pred - target), then average by n.
    """

    def forward(self, predictions: Matrix, targets: Matrix) -> float:
        _check_shape(predictions, targets)
        diff = predictions - targets
        squared = diff * diff
        n = predictions.rows * predictions.cols
        return squared.sum_all() / n

    def backward(self, predictions: Matrix, targets: Matrix) -> Matrix:
        _check_shape(predictions, targets)
        n = predictions.rows * predictions.cols
        return (predictions - targets) * (2.0 / n)


class BinaryCrossEntropy(Loss):
    """Binary Cross-Entropy for binary classification.

    Forward: L = -(1/n) * sum(targets * log(pred) + (1-targets) * log(1-pred))
    Backward: dL/dpred = -(1/n) * (targets/pred - (1-targets)/(1-pred))

    CRITICAL: Clip predictions to range [1e-15, 1 - 1e-15] before
    computing log to prevent log(0) = -infinity.
    """

    def forward(self, predictions: Matrix, targets: Matrix) -> float:
        _check_shape(predictions, targets)
        pred = predictions.clip(_EPS, _EPS_HIGH)
        rows, cols = pred.shape()
        ones = Matrix([[1.0] * cols for _ in range(rows)])
        one_minus_pred = ones - pred
        one_minus_targets = ones - targets
        log_pred = pred.apply(lambda x: math.log(x))
        log_one_minus_pred = one_minus_pred.apply(lambda x: math.log(x))
        term1 = targets * log_pred
        term2 = one_minus_targets * log_one_minus_pred
        n = predictions.rows * predictions.cols
        return -(term1 + term2).sum_all() / n

    def backward(self, predictions: Matrix, targets: Matrix) -> Matrix:
        _check_shape(predictions, targets)
        pred = predictions.clip(_EPS, _EPS_HIGH)
        n = predictions.rows * predictions.cols
        scale = -1.0 / n
        grad_data = []
        for i in range(pred.rows):
            row = []
            for j in range(pred.cols):
                t = targets.data[i][j]
                p = pred.data[i][j]
                # dL/dpred = -(1/n) * (t/p - (1-t)/(1-p))
                row.append(scale * (t / p - (1.0 - t) / (1.0 - p)))
            grad_data.append(row)
        return Matrix(grad_data)


class CategoricalCrossEntropy(Loss):
    """Categorical Cross-Entropy for multi-class classification.

    Expects predictions from Softmax and targets as one-hot encoded.
    Forward: L = -(1/n) * sum over all elements (targets * log(pred))
    Backward: dL/dpred = predictions - targets

    This is the simplified gradient when combined with Softmax output:
    the softmax + cross-entropy gradient simplifies to pred - target.

    CRITICAL: Clip predictions to range [1e-15, 1 - 1e-15] before log.
    n = number of samples (rows).
    """

    def forward(self, predictions: Matrix, targets: Matrix) -> float:
        _check_shape(predictions, targets)
        pred = predictions.clip(_EPS, _EPS_HIGH)
        log_pred = pred.apply(lambda x: math.log(x))
        n = predictions.rows
        return -(targets * log_pred).sum_all() / n

    def backward(self, predictions: Matrix, targets: Matrix) -> Matrix:
        _check_shape(predictions, targets)
        pred = predictions.clip(_EPS, _EPS_HIGH)
        return pred - targets
