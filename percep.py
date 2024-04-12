from itertools import islice

import numpy as np
from numpy import ndarray
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

Batch = tuple[ndarray, ndarray]


def get_data():
    dataset = fetch_california_housing()
    data = dataset.data
    target = dataset.target

    s = StandardScaler()
    data = s.fit_transform(data)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
    return X_train, y_train, X_test, y_test


def loader(X: np.ndarray, y: np.ndarray, batch_size: int) -> Batch:
    X, y = X.copy(), y.copy()

    position = 0
    while True:
        if position >= X.shape[0]:
            position = 0

        if position + batch_size > X.shape[0]:
            bs = X.shape[0] - position
        else:
            bs = batch_size

        X_batch, y_batch = X[position : position + bs], y[position : position + bs]
        yield X_batch, y_batch
        position += bs


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


class Model:
    """
    Two layer perceptron
    X───►┌─┐ F                        @ : dot product
         │@├──┐                       + : addition
    W1──►└─┘  └►┌─┐ E  ┌─┐ D          S : sigmoid function
                │+├───►│S├──┐         - : subtraction
    B1─────────►└─┘    └─┘  └►┌─┐ C   ^2: power of two
                              │@├──┐
    W2───────────────────────►└─┘  └►┌─┐ B
                                     │+├──┐
    B2──────────────────────────────►└─┘  └►┌─┐ A  ┌──┐
                                            │-├───►│^2├─►L
    Y──────────────────────────────────────►└─┘    └──┘
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, lr: float) -> None:
        self.W1 = np.random.randn(in_dim, hidden_dim)
        self.B1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, out_dim)
        self.B2 = np.zeros((1, out_dim))

        self.lr = lr

    def fit_batch(self, X: np.ndarray, Y: np.ndarray) -> float:
        F = np.dot(X, self.W1)  # (bs, hidden_dim)
        E = F + self.B1  # (bs, hidden_dim)
        D = sigmoid(E)  # (bs, hidden_dim)
        C = np.dot(D, self.W2)  # (bs, out_dim)
        B = C + self.B2  # (bs, out_dim)
        A = Y - B  # (bs, out_dim)
        L = np.power(A, 2).mean()  # (1,)

        dLdA = 2 * A  # (bs, out_dim)
        dAdB = -1  # (bs, out_dim)
        dBdC = np.ones_like(C)  # (bs, out_dim)
        dBdb2 = np.ones_like(self.B2)  # (bs, out_dim)
        dCdD = self.W2.T  # (out_dim, hidden_dim)
        dCdw2 = D.T  # (hidden_dim, bs)
        dDdE = D * (1 - D)  # (bs, hidden_dim)
        dEdF = np.ones_like(F)  # (bs, hidden_dim)
        dEdb1 = np.ones_like(self.B1)  # (bs, hidden_dim)
        dFdw1 = X.T  # (in_dim, bs)

        dLdb2 = np.mean(dLdA * dAdB * dBdb2, axis=0, keepdims=True)  # (1, out_dim)
        dLdw2 = np.dot(dCdw2, dLdA * dAdB * dBdC)  # (bs, out_dim)
        dLdb1 = np.mean(
            np.dot(dLdA * dAdB * dBdC, dCdD) * dDdE * dEdb1, axis=0, keepdims=True
        )  # (1, hidden_dim)
        dLdw1 = np.dot(dFdw1, np.dot(dLdA * dAdB * dBdC, dCdD) * dDdE * dEdF)  # (bs, in_dim)

        # update gradients
        self.B2 -= self.lr * dLdb2
        self.W2 -= self.lr * dLdw2
        self.B1 -= self.lr * dLdb1
        self.W1 -= self.lr * dLdw1

        return L

    def predict(self, X: np.ndarray) -> np.ndarray:
        F = np.dot(X, self.W1)  # (bs, hidden_dim)
        E = F + self.B1  # (bs, hidden_dim)
        D = sigmoid(E)  # (bs, hidden_dim)
        C = np.dot(D, self.W2)  # (bs, out_dim)
        B = C + self.B2  # (bs, out_dim)
        return B


if __name__ == "__main__":
    np.random.seed(180708)
    batch_size = 23
    num_steps = 1000
    in_features = 8
    hidden_dim = 20
    out_features = 1
    lr = 0.001

    X_train, y_train, X_test, y_test = get_data()

    model = Model(in_features, hidden_dim, out_features, lr=lr)

    losses = []
    for X, y in islice(loader(X_train, y_train, batch_size), num_steps):
        loss = model.fit_batch(X, y)
        losses.append(loss)

    print(f"Initial loss {losses[0]:.4f}, last loss {losses[-1]:.4f}")
    assert losses[-1] / losses[0] <= 0.05