"""Tests for activations module."""

import unittest
from nn.math_utils import Matrix
from nn.activations import Sigmoid, ReLU, LeakyReLU, Tanh, Softmax, Linear


class TestSigmoid(unittest.TestCase):
    """Test Sigmoid activation."""

    def test_sigmoid_forward_at_zero(self):
        """Sigmoid(0) = 0.5."""
        act = Sigmoid()
        z = Matrix([[0.0]])
        out = act.forward(z)
        self.assertAlmostEqual(out.data[0][0], 0.5, places=6)

    def test_sigmoid_derivative_at_zero(self):
        """Sigmoid derivative at 0 = 0.25 (f'(0) = f(0)*(1-f(0)) = 0.25)."""
        act = Sigmoid()
        z = Matrix([[0.0]])
        d = act.derivative(z)
        self.assertAlmostEqual(d.data[0][0], 0.25, places=6)


class TestReLU(unittest.TestCase):
    """Test ReLU activation."""

    def test_relu_forward_negative_becomes_zero(self):
        """ReLU: negative input -> 0."""
        act = ReLU()
        z = Matrix([[-1.0, -5.0]])
        out = act.forward(z)
        self.assertEqual(out.data[0][0], 0.0)
        self.assertEqual(out.data[0][1], 0.0)

    def test_relu_forward_positive_unchanged(self):
        """ReLU: positive input -> same value."""
        act = ReLU()
        z = Matrix([[1.0, 3.5]])
        out = act.forward(z)
        self.assertEqual(out.data[0][0], 1.0)
        self.assertEqual(out.data[0][1], 3.5)

    def test_relu_derivative_negative_zero(self):
        """ReLU derivative: negative input -> 0."""
        act = ReLU()
        z = Matrix([[-1.0]])
        d = act.derivative(z)
        self.assertEqual(d.data[0][0], 0.0)

    def test_relu_derivative_positive_one(self):
        """ReLU derivative: positive input -> 1."""
        act = ReLU()
        z = Matrix([[1.0]])
        d = act.derivative(z)
        self.assertEqual(d.data[0][0], 1.0)


class TestTanh(unittest.TestCase):
    """Test Tanh activation."""

    def test_tanh_forward_at_zero(self):
        """Tanh(0) = 0."""
        act = Tanh()
        z = Matrix([[0.0]])
        out = act.forward(z)
        self.assertAlmostEqual(out.data[0][0], 0.0, places=6)


class TestLeakyReLU(unittest.TestCase):
    """Test LeakyReLU with custom alpha."""

    def test_leaky_relu_negative_scaled_by_alpha(self):
        """LeakyReLU: negative input -> alpha * input."""
        act = LeakyReLU(alpha=0.1)
        z = Matrix([[-10.0]])
        out = act.forward(z)
        self.assertAlmostEqual(out.data[0][0], -1.0, places=6)

    def test_leaky_relu_derivative_negative_is_alpha(self):
        """LeakyReLU derivative: negative -> alpha."""
        act = LeakyReLU(alpha=0.2)
        z = Matrix([[-5.0]])
        d = act.derivative(z)
        self.assertAlmostEqual(d.data[0][0], 0.2, places=6)


class TestSoftmax(unittest.TestCase):
    """Test Softmax activation."""

    def test_softmax_forward_sums_to_one_per_row(self):
        """Softmax output sums to 1.0 per row."""
        act = Softmax()
        z = Matrix([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
        out = act.forward(z)
        for i in range(out.rows):
            row_sum = sum(out.data[i])
            self.assertAlmostEqual(row_sum, 1.0, places=6)

    def test_softmax_large_values_no_overflow(self):
        """Softmax with large values does not overflow (no inf/nan)."""
        act = Softmax()
        z = Matrix([[100.0, 200.0, 300.0]])
        out = act.forward(z)
        for i in range(out.rows):
            for j in range(out.cols):
                self.assertFalse(__import__("math").isinf(out.data[i][j]))
                self.assertFalse(__import__("math").isnan(out.data[i][j]))
        row_sum = sum(out.data[0])
        self.assertAlmostEqual(row_sum, 1.0, places=5)


class TestLinear(unittest.TestCase):
    """Test Linear (identity) activation."""

    def test_linear_is_identity(self):
        """Linear forward returns input unchanged."""
        act = Linear()
        z = Matrix([[1.0, 2.0], [3.0, 4.0]])
        out = act.forward(z)
        self.assertEqual(out.data, z.data)


if __name__ == "__main__":
    unittest.main()
