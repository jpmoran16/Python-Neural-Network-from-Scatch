"""Binary classification example."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn.activations import ReLU, Sigmoid
from nn.layers import Dense
from nn.losses import BinaryCrossEntropy
from nn.network import Network
from nn.data_utils import generate_circles_data, train_test_split, accuracy_score

# Generate concentric circles dataset
X, y = generate_circles_data(n_samples=200, noise=0.1, seed=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2, seed=42)

# Build network: Dense(2->8, ReLU) -> Dense(8->4, ReLU) -> Dense(4->1, Sigmoid)
net = Network()
net.add(Dense(2, 8, ReLU(), weight_init="he"))
net.add(Dense(8, 4, ReLU(), weight_init="he"))
net.add(Dense(4, 1, Sigmoid(), weight_init="xavier"))
net.set_loss(BinaryCrossEntropy())

# Train for 1000 epochs
net.train(X_train, y_train, epochs=1000, learning_rate=0.1, verbose=True, print_every=200)

# Evaluate on test set
pred_test = net.predict(X_test)
acc_test = accuracy_score(pred_test, y_test)
loss_test = net.loss_function.forward(pred_test, y_test)

# Evaluate on full dataset
pred_all = net.predict(X)
acc_all = accuracy_score(pred_all, y)

print(f"\nTest  Loss:     {loss_test:.6f}")
print(f"Test  Accuracy: {acc_test:.2%}")
print(f"Final Accuracy: {acc_all:.2%}")
