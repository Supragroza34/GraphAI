import numpy as np

#Layers:
# forward() -> input -> output
# backward() - dLoss/dOutput and sends is backward (backpropogation)
# Params() - change based on backward()
# Dense Layer - fully connected layer (tranforms input to output array via weights and biases)


#Equation: Y = XW + b
class Dense:

    def __init__(self, n_in, n_out, rng):
        scale = np.sqrt(2.0/ n_in)
        self.W = rng.normal(0.0, scale, (n_in, n_out))
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x, training=True):
        self.x = x
        return x @ self.W + self.b   #X:[3 4]        *    W:[20]
                                       #[2 4]               [10] + b: 16
                                       #[5 6]               [15]
    def backward(self, grad):
        """
        {dL / dW = X^Transpose *  G}
        """
        self.dW = self.x.T @ grad
        self.db = grad.sum(axis=0) #dL/  db is summing all gradients
        return grad @ self.W.T #chain rule dL/dW * dW / dX

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]

    
class RelU:

    def forward(self,x, training=True):
        self.mask = x > 0
        return np.maximum(x, 0.0)

    def backward(self, grad):
        return grad * self.mask

    def params(self):
        return []



class Dropout:

    def __init__(self, rate, rng):
        self.keep = 1.0 - rate
        self.rng = rng

    def forward(self, x, training=True):
        if not training:
            return x
        kept = self.rng.random(x.shape) < self.keep
        self.mask = kept / self.keep
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask

    def params(self):
        return []


    
    
