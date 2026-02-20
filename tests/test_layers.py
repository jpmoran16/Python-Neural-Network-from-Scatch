"""Tests for layers module."""

import unittest
import math
from nn.math_utils import Matrix
from nn.activations import Sigmoid, ReLU, Linear
from nn.layers import Dense


class TestDenseForward(unittest.TestCase):
    """Test Dense layer forward pass."""

    def test_dense_forward_output_shape(self):
        """Dense forward output shape is (batch_size, output_size)."""
        layer = Dense(4, 3, Sigmoid(), weight_init="random")
        X = Matrix([[1.0] * 4, [2.0] * 4])  # batch 2, features 4
        out = layer.forward(X)
        self.assertEqual(out.shape(), (2, 3))

    def test_dense_forward_known_weights(self):
        """Dense forward with manually set weights gives predictable output."""
        layer = Dense(2, 2, Linear(), weight_init="random")
        layer.weights = Matrix([[1.0, 0.0], [0.0, 1.0]])
        layer.biases = Matrix([[0.0, 0.0]])
        X = Matrix([[1.0, 2.0]])
        out = layer.forward(X)
        # [1,2] @ [[1,0],[0,1]] + [0,0] = [1, 2]
        self.assertAlmostEqual(out.data[0][0], 1.0, places=5)
        self.assertAlmostEqual(out.data[0][1], 2.0, places=5)


class TestDenseBackward(unittest.TestCase):
    """Test Dense layer backward returns correct gradient shapes."""

    def test_dense_backward_returns_correct_shapes(self):
        """Dense backward returns (weight_grad, bias_grad, input_grad) with correct shapes."""
        layer = Dense(4, 3, Sigmoid(), weight_init="random")
        X = Matrix([[1.0] * 4, [2.0] * 4])
        layer.forward(X)
        output_grad = Matrix([[0.1] * 3, [0.1] * 3])
        w_grad, b_grad, in_grad = layer.backward(output_grad)
        self.assertEqual(w_grad.shape(), (4, 3))
        self.assertEqual(b_grad.shape(), (1, 3))
        self.assertEqual(in_grad.shape(), (2, 4))


class TestXavierInitialization(unittest.TestCase):
    """Test Xavier weight initialization."""

    def test_xavier_values_in_expected_range(self):
        """Xavier initialization produces values in expected scaled range."""
        layer = Dense(10, 5, Sigmoid(), weight_init="xavier")
        scale = math.sqrt(2.0 / (10 + 5))
        for i in range(layer.weights.rows):
            for j in range(layer.weights.cols):
                self.assertLessEqual(abs(layer.weights.data[i][j]), scale * 1.01)


class TestHeInitialization(unittest.TestCase):
    """Test He weight initialization."""

    def test_he_values_in_expected_range(self):
        """He initialization produces values in expected scaled range."""
        layer = Dense(10, 5, ReLU(), weight_init="he")
        scale = math.sqrt(2.0 / 10)
        for i in range(layer.weights.rows):
            for j in range(layer.weights.cols):
                self.assertLessEqual(abs(layer.weights.data[i][j]), scale * 1.01)


if __name__ == "__main__":
    unittest.main()
