import numpy as np
import matplotlib.pyplot as plt
from data import load_data, one_hot, batches

data = load_data()
for name, (X,y) in data.items():
    print(name, X.shape, y.shape)

X,y = data["train"]
print(f"Pixel range is {X.min()} to {X.max()}")
print(f"label {y[0]} becomes {one_hot(y[:1])[0]}")

rng  = np.random.default_rng(0)
sizes = [len(xb) for xb, yb in batches(X, y, 32, rng)]

print(f"{len(sizes)} batches the last has {sizes[-1]} pictures")

fig, axes = plt.subplots(2,8, figsize=(8, 2.4))
for ax, picture, label in zip(axes.flat, X, y):
    ax.imshow(picture.reshape(8,8), cmap="gray_r")
    ax.set_title(label)
    ax.axis("off")

fig.savefig("digits.png", dpi=150, bbox_inches="tight")