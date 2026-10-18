import numpy as np
from layers import Dense, RelU, Dropout


rng = np.random.default_rng(0)
dense = Dense(3,2, rng)
dense.W = np.array([[1.0, -1.0], [0.0, -2.0], [0.5, 0.5]]) ##weights
dense.b = np.array([0.0, 1.0])
relu = RelU()

#Y = [3, -2]

x = np.array([[1.0,2.0,4.0]]) #one picture 3 numbers
hidden = dense.forward(x) ##hrough dense layer
out = relu.foward(hidden) #through relu layer -> Y  = [3,0]
print(f"dense out {hidden}")
print(f"relu out {out}")

grad = relu.backward(np.array([[1.0,1.0]])) #-> grad [1,0]
print(f"after relu backward {grad}")
print(f"returned to input {dense.backward(grad)}")
print(f"dW: {dense.dW.tolist()}")
print(f"db: {dense.db}")

drop = Dropout(0.2, rng)
ones = np.ones((1, 10000)) # 10,000 ones, to count what gets dropped
dropped = drop.forward(ones)
print("zeroed:", np.mean(dropped == 0), " mean:", dropped.mean())
print("untouched when not training:", np.array_equal(drop.forward(ones, training=False), ones))



