from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropy
from sklearn import datasets
import matplotlib.pyplot as plt
import random

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10
GRAD_TYPE_DEFAULT = 'BATCH'

FLAGS = None

x, t = datasets.make_moons(n_samples=2000, shuffle=True, noise=None, random_state=None)
t = np.array([[t[i], 1-t[i]] for i in range(len(t))])
train_x = x[:1400]
train_t = t[:1400]
test_x = x[1400:]
test_t = t[1400:]

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    cnt = 0
    for i in range(len(predictions)):
        if (np.around(predictions[i]) == targets[i]).all():
            cnt += 1
    accu = cnt / len(predictions)
    # print(str(cnt))
    return accu


def cal_accuracy(mlp, data, label):
    loss = 0
    predictions = []
    for i in range(len(label)):
        data_f = data[i].reshape(1, -1)
        label_f = label[i].reshape(1, -1)
        out = mlp.forward(data_f)
        loss += CrossEntropy().forward(out, label_f)
        predictions.append(out)
    return accuracy(predictions, label), loss/len(data)


def train(mlp, FLAGS):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    accu_train = []
    accu_test = []
    loss_train = []
    loss_test = []
    x_axis = []
    for freq in range(FLAGS.max_steps):
        for i in range(len(train_x)):
            data = train_x[i].reshape(1, -1)
            label = train_t[i].reshape(1, -1)
            out = mlp.forward(data)
            dout = out-label
            mlp.backward(dout, FLAGS.grad_type)
        if FLAGS.grad_type == 'BATCH':
            mlp.train(len(train_x))

        if freq % FLAGS.eval_freq == 0:
            x_axis.append(freq)
            accu, loss = cal_accuracy(mlp, train_x, train_t)
            accu_train.append(accu)
            loss_train.append(loss)
            if freq%100 == 0:
                print('epoch: '+ str(freq)+'/1500')
                print('train: accu: '+str(format(accu, '.3f'))+', loss: '+str(format(loss, '.3f')))

            # test
            accu, loss = cal_accuracy(mlp, test_x, test_t)
            accu_test.append(accu)
            loss_test.append(loss)
            if freq%100 == 0:
                print('test: accu: '+str(format(accu, '.3f'))+', loss: '+str(format(loss, '.3f')))
    # plot_graph(x_axis, accu_train, accu_test, loss_train, loss_test)
    return x_axis, accu_train, accu_test, loss_train, loss_test

def plot_graph(x_axis, accu_train, accu_test, loss_train, loss_test):
    fig_accu = plt.subplot(2, 1, 1)
    # y_major_locator = MultipleLocator(10)
    # ax = plt.gca()
    # ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(0, 1.1)
    fig_loss = plt.subplot(2, 1, 2)
    fig_accu.plot(x_axis, accu_train, c='red', label='training accuracy')
    fig_accu.plot(x_axis, accu_test, c='blue', label='test accuracy')
    fig_accu.legend()
    fig_loss.plot(x_axis, loss_train, c='green', label='train loss')
    fig_loss.plot(x_axis, loss_test, c='yellow', label='test loss')
    fig_loss.legend()
    plt.show()

def main():
    """
    Main function
    """
    n_input = 2
    n_hidden = list(map(int, FLAGS.dnn_hidden_units.split()))
    n_classes = 2
    # print(train_t)
    mlp = MLP(n_input, n_hidden, n_classes, FLAGS.learning_rate)
    train(mlp, FLAGS)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--grad_type', type=str, default=GRAD_TYPE_DEFAULT,
                        help='Choose whether to use batch gradient descent or stochastic gradient descent')
    FLAGS, unparsed = parser.parse_known_args()
    main()
