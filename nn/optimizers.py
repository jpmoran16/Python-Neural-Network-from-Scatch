"""Optimization algorithms for training neural networks.

Uses only math and Matrix from math_utils.
Optimizers apply update rules to layer weights and biases using gradients
computed by the layer's backward pass.
"""

import math
from nn.math_utils import Matrix


class Optimizer:
    """Base optimizer class."""

    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    def step(
        self,
        layer: "Dense",
        weight_grad: Matrix,
        bias_grad: Matrix,
    ) -> None:
        """Apply one update step to the layer using the given gradients.
        Subclasses must override."""
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent.

    weights -= lr * gradient
    Optional momentum support:
      velocity = momentum * velocity - lr * gradient
      weights += velocity
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0) -> None:
        super().__init__(learning_rate)
        self.momentum = momentum
        self._velocity_w: dict[int, Matrix] = {}
        self._velocity_b: dict[int, Matrix] = {}

    def step(
        self,
        layer: "Dense",
        weight_grad: Matrix,
        bias_grad: Matrix,
    ) -> None:
        lid = id(layer)
        if self.momentum == 0.0:
            layer.weights = layer.weights - (weight_grad * self.learning_rate)
            layer.biases = layer.biases - (bias_grad * self.learning_rate)
            return
        if lid not in self._velocity_w:
            self._velocity_w[lid] = Matrix.zeros(layer.weights.rows, layer.weights.cols)
            self._velocity_b[lid] = Matrix.zeros(layer.biases.rows, layer.biases.cols)
        v_w = self._velocity_w[lid]
        v_b = self._velocity_b[lid]
        v_w = (v_w * self.momentum) - (weight_grad * self.learning_rate)
        v_b = (v_b * self.momentum) - (bias_grad * self.learning_rate)
        self._velocity_w[lid] = v_w
        self._velocity_b[lid] = v_b
        layer.weights = layer.weights + v_w
        layer.biases = layer.biases + v_b


class Adam(Optimizer):
    """Adam optimizer.

    Maintains per-parameter:
      m (first moment / mean of gradients)
      v (second moment / mean of squared gradients)

    Update rules:
      m = beta1 * m + (1 - beta1) * gradient
      v = beta2 * v + (1 - beta2) * gradient^2
      m_hat = m / (1 - beta1^t)  (bias correction)
      v_hat = v / (1 - beta2^t)  (bias correction)
      weights -= lr * m_hat / (sqrt(v_hat) + epsilon)

    Default: beta1=0.9, beta2=0.999, epsilon=1e-8
    Track timestep t (incremented each step call).
    Initialize m and v as zero matrices on first step for each layer.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._t = 0
        self._m_w: dict[int, Matrix] = {}
        self._m_b: dict[int, Matrix] = {}
        self._v_w: dict[int, Matrix] = {}
        self._v_b: dict[int, Matrix] = {}

    def step(
        self,
        layer: "Dense",
        weight_grad: Matrix,
        bias_grad: Matrix,
    ) -> None:
        self._t += 1
        lid = id(layer)
        if lid not in self._m_w:
            self._m_w[lid] = Matrix.zeros(layer.weights.rows, layer.weights.cols)
            self._m_b[lid] = Matrix.zeros(layer.biases.rows, layer.biases.cols)
            self._v_w[lid] = Matrix.zeros(layer.weights.rows, layer.weights.cols)
            self._v_b[lid] = Matrix.zeros(layer.biases.rows, layer.biases.cols)
        m_w = self._m_w[lid]
        m_b = self._m_b[lid]
        v_w = self._v_w[lid]
        v_b = self._v_b[lid]

        m_w = (m_w * self.beta1) + (weight_grad * (1.0 - self.beta1))
        m_b = (m_b * self.beta1) + (bias_grad * (1.0 - self.beta1))
        grad_sq_w = weight_grad * weight_grad
        grad_sq_b = bias_grad * bias_grad
        v_w = (v_w * self.beta2) + (grad_sq_w * (1.0 - self.beta2))
        v_b = (v_b * self.beta2) + (grad_sq_b * (1.0 - self.beta2))

        self._m_w[lid] = m_w
        self._m_b[lid] = m_b
        self._v_w[lid] = v_w
        self._v_b[lid] = v_b

        bias1 = 1.0 - (self.beta1 ** self._t)
        bias2 = 1.0 - (self.beta2 ** self._t)
        m_hat_w = m_w * (1.0 / bias1)
        m_hat_b = m_b * (1.0 / bias1)
        v_hat_w = v_w * (1.0 / bias2)
        v_hat_b = v_b * (1.0 / bias2)

        def adam_update(m_hat: Matrix, v_hat: Matrix) -> Matrix:
            rows, cols = m_hat.shape()
            out_data = []
            for i in range(rows):
                row = []
                for j in range(cols):
                    denom = math.sqrt(v_hat.data[i][j]) + self.epsilon
                    row.append(self.learning_rate * m_hat.data[i][j] / denom)
                out_data.append(row)
            return Matrix(out_data)

        update_w = adam_update(m_hat_w, v_hat_w)
        update_b = adam_update(m_hat_b, v_hat_b)
        layer.weights = layer.weights - update_w
        layer.biases = layer.biases - update_b


class RMSProp(Optimizer):
    """RMSProp optimizer.

    Maintains per-parameter:
      cache (running average of squared gradients)

    Update rules:
      cache = decay_rate * cache + (1 - decay_rate) * gradient^2
      weights -= lr * gradient / (sqrt(cache) + epsilon)

    Default: decay_rate=0.9, epsilon=1e-8
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        decay_rate: float = 0.9,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self._cache_w: dict[int, Matrix] = {}
        self._cache_b: dict[int, Matrix] = {}

    def step(
        self,
        layer: "Dense",
        weight_grad: Matrix,
        bias_grad: Matrix,
    ) -> None:
        lid = id(layer)
        if lid not in self._cache_w:
            self._cache_w[lid] = Matrix.zeros(layer.weights.rows, layer.weights.cols)
            self._cache_b[lid] = Matrix.zeros(layer.biases.rows, layer.biases.cols)
        cache_w = self._cache_w[lid]
        cache_b = self._cache_b[lid]
        grad_sq_w = weight_grad * weight_grad
        grad_sq_b = bias_grad * bias_grad
        cache_w = (cache_w * self.decay_rate) + (grad_sq_w * (1.0 - self.decay_rate))
        cache_b = (cache_b * self.decay_rate) + (grad_sq_b * (1.0 - self.decay_rate))
        self._cache_w[lid] = cache_w
        self._cache_b[lid] = cache_b

        def rms_update(grad: Matrix, cache: Matrix) -> Matrix:
            rows, cols = grad.shape()
            out_data = []
            for i in range(rows):
                row = []
                for j in range(cols):
                    denom = math.sqrt(cache.data[i][j]) + self.epsilon
                    row.append(self.learning_rate * grad.data[i][j] / denom)
                out_data.append(row)
            return Matrix(out_data)

        update_w = rms_update(weight_grad, cache_w)
        update_b = rms_update(bias_grad, cache_b)
        layer.weights = layer.weights - update_w
        layer.biases = layer.biases - update_b
