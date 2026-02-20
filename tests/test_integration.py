"""Integration tests: full end-to-end training and evaluation."""

import os
import sys
import unittest
from nn.math_utils import Matrix, one_hot_encode
from nn.activations import Sigmoid, ReLU, Softmax
from nn.losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy
from nn.layers import Dense
from nn.network import Network
from nn.optimizers import SGD, Adam
from nn.data_utils import (
    generate_xor_data,
    generate_circles_data,
    generate_spiral_data,
    accuracy_score,
)


class TestXOREndToEnd(unittest.TestCase):
    """Full end-to-end: XOR problem."""

    def test_xor_converges_low_loss(self):
        """XOR: Dense(2,4,Sigmoid)->Dense(4,1,Sigmoid), 5000 epochs, final loss < 0.1."""
        X, y = generate_xor_data()
        net = Network()
        net.add(Dense(2, 4, Sigmoid(), weight_init="xavier"))
        net.add(Dense(4, 1, Sigmoid(), weight_init="xavier"))
        net.set_loss(MSE())
        net.train(X, y, epochs=5000, learning_rate=0.5, verbose=False)
        loss = net.loss_function.forward(net.predict(X), y)
        self.assertLess(loss, 0.1, msg=f"XOR final loss {loss} should be < 0.1")

    def test_xor_predictions_close_to_expected(self):
        """XOR: predictions within 0.3 of expected [0,1,1,0]."""
        X, y = generate_xor_data()
        net = Network()
        net.add(Dense(2, 4, Sigmoid(), weight_init="xavier"))
        net.add(Dense(4, 1, Sigmoid(), weight_init="xavier"))
        net.set_loss(MSE())
        net.train(X, y, epochs=5000, learning_rate=0.5, verbose=False)
        pred = net.predict(X)
        expected = [0.0, 1.0, 1.0, 0.0]
        for i in range(4):
            self.assertAlmostEqual(
                pred.data[i][0],
                expected[i],
                delta=0.3,
                msg=f"Sample {i}: expected ~{expected[i]}, got {pred.data[i][0]}",
            )


class TestBinaryClassificationCircles(unittest.TestCase):
    """Full end-to-end: binary classification on circles data."""

    def test_circles_accuracy_above_80(self):
        """Circles: Dense(2,8,ReLU)->Dense(8,4,ReLU)->Dense(4,1,Sigmoid), 1000 epochs, accuracy > 80%."""
        X, y = generate_circles_data(n_samples=200, noise=0.1, seed=42)
        net = Network()
        net.add(Dense(2, 8, ReLU(), weight_init="he"))
        net.add(Dense(8, 4, ReLU(), weight_init="he"))
        net.add(Dense(4, 1, Sigmoid(), weight_init="xavier"))
        net.set_loss(BinaryCrossEntropy())
        net.train(X, y, epochs=1000, learning_rate=0.1, verbose=False)
        pred = net.predict(X)
        acc = accuracy_score(pred, y)
        self.assertGreater(
            acc, 0.80, msg=f"Circles accuracy {acc:.2%} should be > 80%"
        )


class TestMulticlassSpiral(unittest.TestCase):
    """Full end-to-end: multi-class classification on spiral data."""

    def test_spiral_accuracy_above_60(self):
        """Spiral: Dense(2,16,ReLU)->Dense(16,8,ReLU)->Dense(8,3,Softmax), 2000 epochs with Adam, accuracy > 60%."""
        X, y_labels = generate_spiral_data(samples_per_class=50, num_classes=3, seed=42)
        y = one_hot_encode(y_labels, 3)
        net = Network()
        net.add(Dense(2, 16, ReLU(), weight_init="he"))
        net.add(Dense(16, 8, ReLU(), weight_init="he"))
        net.add(Dense(8, 3, Softmax(), weight_init="xavier"))
        net.set_loss(CategoricalCrossEntropy())
        net.train(
            X, y, epochs=2000, learning_rate=0.1,
            optimizer=Adam(learning_rate=0.01), verbose=False
        )
        pred = net.predict(X)
        acc = accuracy_score(pred, y)
        self.assertGreater(
            acc, 0.60, msg=f"Spiral accuracy {acc:.2%} should be > 60%"
        )


class TestAdamVsSGD(unittest.TestCase):
    """Adam converges faster than SGD on same setup."""

    def test_adam_converges_faster_than_sgd(self):
        """Same architecture and data: at epoch 500, Adam has lower loss than SGD."""
        X, y_labels = generate_spiral_data(samples_per_class=30, num_classes=3, seed=123)
        y = one_hot_encode(y_labels, 3)
        lr = 0.1

        net_sgd = Network()
        net_sgd.add(Dense(2, 16, ReLU(), weight_init="he"))
        net_sgd.add(Dense(16, 8, ReLU(), weight_init="he"))
        net_sgd.add(Dense(8, 3, Softmax(), weight_init="xavier"))
        net_sgd.set_loss(CategoricalCrossEntropy())
        net_sgd.train(
            X, y, epochs=500, learning_rate=lr, optimizer=SGD(learning_rate=lr), verbose=False
        )
        loss_sgd = net_sgd.loss_function.forward(net_sgd.predict(X), y)

        net_adam = Network()
        net_adam.add(Dense(2, 16, ReLU(), weight_init="he"))
        net_adam.add(Dense(16, 8, ReLU(), weight_init="he"))
        net_adam.add(Dense(8, 3, Softmax(), weight_init="xavier"))
        net_adam.set_loss(CategoricalCrossEntropy())
        net_adam.train(
            X, y, epochs=500, learning_rate=lr, optimizer=Adam(learning_rate=lr), verbose=False
        )
        loss_adam = net_adam.loss_function.forward(net_adam.predict(X), y)

        self.assertLess(
            loss_adam,
            loss_sgd,
            msg=f"Adam loss {loss_adam:.6f} should be < SGD loss {loss_sgd:.6f} at epoch 500",
        )


def run_and_print_summary():
    """Run all tests and print a pass/fail summary."""
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("SUMMARY: All tests PASSED.")
    else:
        print("SUMMARY: Some tests FAILED.")
        print(f"  Failures: {len(result.failures)}")
        print(f"  Errors:   {len(result.errors)}")
    print("=" * 60)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    # Run full suite and print pass/fail summary
    sys.exit(run_and_print_summary())
