"""Tests for network module."""

import unittest
import os
import tempfile
from nn.math_utils import Matrix, one_hot_encode
from nn.activations import Sigmoid, ReLU, Softmax
from nn.losses import CategoricalCrossEntropy, MSE
from nn.layers import Dense
from nn.network import Network


class TestNetworkAdd(unittest.TestCase):
    """Test Network.add() builds layers correctly."""

    def test_add_builds_layers_correctly(self):
        """Adding layers updates the layers list with correct sizes."""
        net = Network()
        self.assertEqual(len(net.layers), 0)
        net.add(Dense(2, 4, Sigmoid()))
        net.add(Dense(4, 3, Softmax()))
        self.assertEqual(len(net.layers), 2)
        self.assertEqual(net.layers[0].weights.rows, 2)
        self.assertEqual(net.layers[0].weights.cols, 4)
        self.assertEqual(net.layers[1].weights.rows, 4)
        self.assertEqual(net.layers[1].weights.cols, 3)


class TestNetworkSummary(unittest.TestCase):
    """Test Network.summary() runs without error."""

    def test_summary_runs_without_error(self):
        """summary() prints and does not raise."""
        net = Network()
        net.add(Dense(2, 4, ReLU()))
        net.add(Dense(4, 1, Sigmoid()))
        try:
            net.summary()
        except Exception as e:
            self.fail(f"summary() raised {e}")


class TestNetworkForward(unittest.TestCase):
    """Test Network.forward() produces correct output shape."""

    def test_forward_produces_correct_output_shape(self):
        """Forward pass output shape is (batch_size, output_size)."""
        net = Network()
        net.add(Dense(3, 5, ReLU()))
        net.add(Dense(5, 2, Softmax()))
        X = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = net.forward(X)
        self.assertEqual(out.shape(), (2, 2))


class TestNetworkSaveLoad(unittest.TestCase):
    """Test Network.save() and Network.load() round-trip."""

    def test_save_and_load_round_trip(self):
        """Saved model can be loaded and produces same predictions."""
        net = Network()
        net.add(Dense(2, 4, Sigmoid()))
        net.add(Dense(4, 2, Softmax()))
        net.set_loss(CategoricalCrossEntropy())
        X = Matrix([[0.0, 0.0], [1.0, 1.0]])
        y = one_hot_encode([0, 1], 2)
        net.train(X, y, epochs=10, learning_rate=0.3, verbose=False)
        pred_before = net.predict(X)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            net.save(path)
            net2 = Network()
            net2.load(path)
            net2.set_loss(CategoricalCrossEntropy())
            pred_after = net2.predict(X)
            self.assertEqual(pred_before.shape(), pred_after.shape())
            for i in range(pred_before.rows):
                for j in range(pred_before.cols):
                    self.assertAlmostEqual(
                        pred_before.data[i][j], pred_after.data[i][j], places=5
                    )
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestNetworkTrainingReducesLoss(unittest.TestCase):
    """Test that training reduces loss over epochs on simple data."""

    def test_training_reduces_loss(self):
        """Training on simple data reduces loss over epochs."""
        net = Network()
        net.add(Dense(2, 4, Sigmoid()))
        net.add(Dense(4, 1, Sigmoid()))
        net.set_loss(MSE())
        X = Matrix([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = Matrix([[0.0], [1.0], [1.0], [0.0]])
        hist = net.train(X, y, epochs=100, learning_rate=0.5, verbose=False)
        self.assertGreater(len(hist), 0)
        initial_loss = hist[0]["loss"]
        final_loss = hist[-1]["loss"]
        self.assertLess(final_loss, initial_loss)


if __name__ == "__main__":
    unittest.main()
