"""XOR problem example."""

import sys
import os
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

random.seed(42)

from nn.activations import Sigmoid
from nn.layers import Dense
from nn.losses import MSE
from nn.network import Network
from nn.data_utils import generate_xor_data, accuracy_score

# Generate XOR dataset
X, y = generate_xor_data()

# Build network: Dense(2->4, Sigmoid) -> Dense(4->1, Sigmoid)
net = Network()
net.add(Dense(2, 4, Sigmoid(), weight_init="xavier"))
net.add(Dense(4, 1, Sigmoid(), weight_init="xavier"))
net.set_loss(MSE())

# Train for 5000 epochs
net.train(X, y, epochs=5000, learning_rate=0.5, verbose=True, print_every=1000)

# Evaluate
pred = net.predict(X)
acc = accuracy_score(pred, y)
final_loss = net.loss_function.forward(pred, y)

print("\nXOR Predictions:")
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
expected = [0, 1, 1, 0]
for i in range(4):
    raw = pred.data[i][0]
    label = round(raw)
    print(f"  Input {inputs[i]} -> {raw:.4f} (predicted: {label}, expected: {expected[i]})")

print(f"\nFinal Loss:     {final_loss:.6f}")
print(f"Final Accuracy: {acc:.2%}")
