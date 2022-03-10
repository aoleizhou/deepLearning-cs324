import numpy as np


class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Module initialisation.
        Args:
            in_features: input dimension
            out_features: output dimension
        TODO:
        1) Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
        std = 0.0001.
        2) Initialize biases self.params['bias'] with 0. 
        3) Initialize gradients with zeros.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.params = {'weight': np.random.normal(0, 10, (in_features, out_features)),
                       'bias': np.zeros((1, out_features))}
        self.grads = {'weight': np.zeros((in_features, out_features)),
                      'bias': np.zeros((1, out_features))}
        self.x = None

    def forward(self, x):
        """
        Forward pass (i.e., compute output from input).
        Args:
            x: input to the module
        Returns:
            out: output of the module
        Hint: Similarly to pytorch, you can store the computed values inside the object
        and use them in the backward pass computation. This is true for *all* forward methods of *all* modules in this class
        """

        self.x = x
        return np.dot(x, self.params['weight']) + self.params['bias']

    def backward(self, dout):
        """
        Backward pass (i.e., compute gradient).
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to 
        layer parameters in self.grads['weight'] and self.grads['bias']. 
        """

        self.grads['bias'] += dout
        self.grads['weight'] += np.dot(self.x.T, dout)
        dx = np.dot(dout, self.params['weight'].T)
        return dx

    def train(self, rate, size):
        self.params['weight'] = self.params['weight'] - rate * self.grads['weight'] / size
        self.params['bias'] = self.params['bias'] - rate * self.grads['bias'] / size
        self.grads = {'weight': np.zeros((self.in_features, self.out_features)),
                      'bias': np.zeros((1, self.out_features))}


class ReLU(object):
    def __init__(self):
        self.x = None

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
        """
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        """
        Backward pass.
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        result = np.zeros((1, len(dout[0])))
        for i in range(len(dout[0])):
            if self.x[0][i] > 0:
                result[0][i] = self.x[0][i]
        return np.where(self.x > 0, dout, 0)

class SoftMax(object):
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
    
        TODO:
        Implement forward pass of the module. 
        To stabilize computation you should use the so-called Max Trick
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        
        """
        b = np.max(x)
        y = np.exp(x - b)
        return y / y.sum()

    def backward(self, dout):
        """
        Backward pass. 
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        return dout


class CrossEntropy(object):
    def forward(self, x, y):
        """
        Forward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            out: cross entropy loss
        """
        return -np.sum(y * np.log(x+1e-5))

    def backward(self, x, y):
        """
        Backward pass.
        Args:
            x: input to the module out
            y: labels of the input
        Returns:
            dx: gradient of the loss with respect to the input x.
        """
        # result = np.zeros((1, len(y)))
        # for i in range(len(y)):
        #     if x[0][i] != 0:
        #         result[0][i] = (-y[i] / x[0][i])
        #     else:
        #         result[0][i] = 0
        return  -y/x
