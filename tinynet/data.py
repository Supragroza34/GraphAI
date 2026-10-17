import numpy as np
from sklearn.datasets import load_digits

def one_hot(labels, num_classes=10):
    # 3 -> [0,0,0,1,0,0,0,0,0,0]
    out = np.zeros((len(labels), num_classes), dtype=int)
    out[np.arange(len(labels)), labels] = 1
    return out

def load_data(seed=0):
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    X, y = X[order], y[order]

    n_train = int(0.70 * len(X))
    n_val = int(0.15* len(X))
    end_val = n_train + n_val
    return{
        "train": (X[:n_train], y[:n_train]),
        "val": (X[n_train:end_val], y[n_train:end_val]),
        "test": (X[end_val:], y[end_val:]),
    }

def batches(X, y, batch_size, rng):
    order = rng.permutation(len(X))

    for start in range(0,len(X), batch_size):
        idx = order[start:start+batch_size]
        yield X[idx], y[idx]


