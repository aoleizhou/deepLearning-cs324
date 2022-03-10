from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):

    def __init__(self, n_inputs, n_hidden, n_classes, rate):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        self.n_inputs = int(n_inputs)
        self.n_hidden = n_hidden
        self.n_classes = int(n_classes)
        self.rate = rate
        n_unit = [self.n_inputs] + self.n_hidden + [self.n_classes]
        self.layers = {'linear': [Linear(n_unit[i], n_unit[i+1]) for i in range(len(n_hidden)+1)],
                       'relu': [ReLU() for i in range(len(n_hidden))]}

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        for i in range(len(self.n_hidden)):
            x = self.layers['relu'][i].forward(self.layers['linear'][i].forward(x))
        out = SoftMax().forward(self.layers['linear'][-1].forward(x))
        return out

    def backward(self, dout, grad_type):
        """
        Performs backward propagation pass given the loss gradients. 
        Args:
            dout: gradients of the loss
        """
        dx = SoftMax().backward(dout)
        dx = self.layers['linear'][-1].backward(dx)
        if grad_type != 'BATCH':
            self.layers['linear'][-1].train(self.rate, 1)
        for i in range(len(self.layers['relu'])-1, -1, -1):
            dx = self.layers['relu'][i].backward(dx)
            dx = self.layers['linear'][i].backward(dx)
            if grad_type != 'BATCH':
                self.layers['linear'][i].train(self.rate, 1)
        return

    def train(self, size):
        for linear in self.layers['linear']:
            linear.train(self.rate, size)
        return
