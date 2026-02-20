"""Multiclass classification example."""

import sys
import os
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

random.seed(42)

from nn.activations import ReLU, Softmax
from nn.layers import Dense
from nn.losses import CategoricalCrossEntropy
from nn.math_utils import one_hot_encode
from nn.network import Network
from nn.optimizers import Adam
from nn.data_utils import generate_spiral_data, accuracy_score

# Generate spiral dataset (3 classes, 100 samples per class = 300 total)
X, y_labels = generate_spiral_data(samples_per_class=100, num_classes=3, seed=42)
y = one_hot_encode(y_labels, 3)

# Build network: Dense(2->16, ReLU) -> Dense(16->8, ReLU) -> Dense(8->3, Softmax)
net = Network()
net.add(Dense(2, 16, ReLU(), weight_init="he"))
net.add(Dense(16, 8, ReLU(), weight_init="he"))
net.add(Dense(8, 3, Softmax(), weight_init="xavier"))
net.set_loss(CategoricalCrossEntropy())

# Train for 3000 epochs using Adam optimizer
net.train(
    X, y,
    epochs=3000,
    learning_rate=0.1,
    optimizer=Adam(learning_rate=0.01),
    verbose=True,
    print_every=500,
)

# Evaluate on full dataset
pred = net.predict(X)
final_loss = net.loss_function.forward(pred, y)
acc = accuracy_score(pred, y)

print(f"\nFinal Loss:     {final_loss:.6f}")
print(f"Final Accuracy: {acc:.2%}")
