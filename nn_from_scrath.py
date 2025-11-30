import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


# =========================
# Data loading
# =========================

def load_mnist(normalize=True):
    """
    Load the MNIST dataset from OpenML.
    Returns:
        X_train: (60000, 784)
        y_train: (60000,)
        X_test:  (10000, 784)
        y_test:  (10000,)
    """
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist['data'].astype(np.float32)
    y = mnist['target'].astype(int)

    # Standard train/test split for MNIST
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    if normalize:
        X_train /= 255.0
        X_test  /= 255.0

    return X_train, y_train, X_test, y_test


# =========================
# Helper functions
# =========================

def one_hot(y, num_classes=10):
    """
    y: shape (N,) integer labels
    returns: shape (N, num_classes) one-hot vectors
    """
    
    N = y.shape[0]
    oh = np.zeros((N, num_classes))
    oh[np.arange(N), y] = 1
    
    return oh




def accuracy(y_true, y_pred):
    """
    Compute the fraction of correct predictions.
    y_true: (N,)
    y_pred: (N,)
    """

    return np.mean(y_true == y_pred)





def iterate_minibatches(X, y, batch_size, shuffle=True):
    """
    Generator that yields mini-batches.
    """
    assert X.shape[0] == y.shape[0]
    N = X.shape[0]
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, N, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]


# =========================
# Layers
# =========================

class Dense:
    def __init__(self, in_features, out_features):
        """
        Fully connected layer: z = xW + b
        """
        limit = np.sqrt(6 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (in_features, out_features)).astype(np.float32)
        self.b = np.zeros((1, out_features), dtype=np.float32)

        self.x = None    # cache for backprop
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        x: (N, in_features)
        returns: (N, out_features)
        """
        self.x = x
        # TODO: implement z = xW + b
        # hint: use np.dot(x, self.W) + self.b
        raise NotImplementedError("Implement Dense.forward.")

    def backward(self, grad_output):
        """
        grad_output: dL/dz, shape (N, out_features)
        returns: dL/dx, shape (N, in_features)
        """
        # TODO:
        # dW = x^T * grad_output
        # db = sum over batch of grad_output
        # dx = grad_output * W^T
        raise NotImplementedError("Implement Dense.backward.")

    def step(self, lr):
        """
        SGD update.
        """
        self.W -= lr * self.dW
        self.b -= lr * self.db


class ReLU:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        # TODO: return elementwise max(0, x)
        raise NotImplementedError("Implement ReLU.forward.")

    def backward(self, grad_output):
        # TODO: pass gradient where x > 0, else 0
        raise NotImplementedError("Implement ReLU.backward.")


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        # TODO: compute sigmoid(x) = 1/(1+exp(-x)) and store as self.out
        raise NotImplementedError("Implement Sigmoid.forward.")

    def backward(self, grad_output):
        # TODO: derivative of sigmoid is s * (1 - s), where s = self.out
        raise NotImplementedError("Implement Sigmoid.backward.")


# =========================
# Softmax + Cross-Entropy
# =========================

def softmax(logits):
    """
    logits: (N, num_classes)
    returns: probs (N, num_classes)
    """
    # TODO: implement numerically stable softmax
    # 1. subtract max per row
    # 2. exponentiate
    # 3. divide by row-wise sum
    raise NotImplementedError("Implement softmax function.")


def cross_entropy_loss(probs, y_onehot):
    """
    probs: (N, num_classes)
    y_onehot: (N, num_classes)
    returns: scalar loss
    """
    N = probs.shape[0]
    # TODO: compute -sum(y * log(p + eps))/N
    # eps = 1e-9 to avoid log(0)
    raise NotImplementedError("Implement cross_entropy_loss.")


def softmax_cross_entropy_with_logits(logits, y_onehot):
    """
    Combines softmax + cross-entropy and returns:
        loss (scalar)
        grad_logits (N, num_classes) = dL/dlogits
    """
    probs = softmax(logits)
    loss = cross_entropy_loss(probs, y_onehot)

    N = logits.shape[0]
    # TODO: gradient wrt logits: (probs - y_onehot)/N
    raise NotImplementedError("Implement grad_logits in softmax_cross_entropy_with_logits.")


# =========================
# Neural Network container
# =========================

class NeuralNet:
    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10, activation="relu"):
        """
        Simple 2-layer network: input -> Dense -> Activation -> Dense -> logits
        """
        if activation not in ("relu", "sigmoid"):
            raise ValueError("activation must be 'relu' or 'sigmoid'")

        if activation == "relu":
            act_layer = ReLU()
        else:
            act_layer = Sigmoid()

        self.layers = [
            Dense(input_dim, hidden_dim),
            act_layer,
            Dense(hidden_dim, num_classes)
        ]

    def forward(self, x):
        """
        Forward pass through all layers.
        returns logits (N, num_classes)
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_output):
        """
        Backward pass through all layers (reverse order).
        """
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, lr):
        """
        Apply SGD step to all Dense layers.
        """
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.step(lr)


# =========================
# Training loop
# =========================

def train(model, X_train, y_train, X_test, y_test,
          epochs=10, batch_size=64, lr=0.1):
    """
    Train the model with plain SGD.
    """
    for epoch in range(1, epochs + 1):
        # ---- Training ----
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for X_batch, y_batch in iterate_minibatches(X_train, y_train, batch_size):
            y_batch_oh = one_hot(y_batch, num_classes=10)

            # Forward
            logits = model.forward(X_batch)

            # Loss + gradient
            loss, grad_logits = softmax_cross_entropy_with_logits(logits, y_batch_oh)
            total_loss += loss * X_batch.shape[0]

            # Accuracy on this batch
            probs = softmax(logits)
            preds = np.argmax(probs, axis=1)
            total_correct += np.sum(preds == y_batch)
            total_samples += X_batch.shape[0]

            # Backward + step
            model.backward(grad_logits)
            model.step(lr)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # ---- Evaluation on test set ----
        test_logits = model.forward(X_test)
        test_probs = softmax(test_logits)
        test_preds = np.argmax(test_probs, axis=1)
        test_acc = accuracy(y_test, test_preds)

        print(f"Epoch {epoch:02d}: "
              f"loss={train_loss:.4f}  "
              f"train_acc={train_acc:.4f}  "
              f"test_acc={test_acc:.4f}")


# =========================
# Main / entry point
# =========================

def main():
    # Load data
    X_train, y_train, X_test, y_test = load_mnist()
    print("Train:", X_train.shape, y_train.shape)
    print("Test :", X_test.shape, y_test.shape)

    # Optional: visualize one example
    idx = 0
    img = X_train[idx].reshape(28, 28)
    plt.imshow(img, cmap="gray")
    plt.title(f"Label = {y_train[idx]}")
    plt.show()

    # Build model
    model = NeuralNet(
        input_dim=784,
        hidden_dim=128,
        num_classes=10,
        activation="relu"   # or "sigmoid"
    )

    # Train
    train(
        model,
        X_train, y_train,
        X_test, y_test,
        epochs=10,
        batch_size=64,
        lr=0.1
    )


if __name__ == "__main__":
    main()
