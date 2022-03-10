import numpy as np


class Perceptron(object):

    def __init__(self, n_inputs, max_epochs=1e2, learning_rate=1e-2):
        """
        Initializes perceptron object.
        Args:
            n_inputs: number of inputs.
            max_epochs: maximum number of training cycles.
            learning_rate: magnitude of weight changes at each training cycle
            weight: the weight to be trained
        """
        self.n_inputs = n_inputs
        self.max_epochs = int(max_epochs)
        self.learning_rate = learning_rate
        self.weight = np.zeros(2)

    def forward(self, input):
        """
        Predict label from input 
        Args:
            input: array of dimension equal to n_inputs.
        """
        label = np.sign(np.dot(self.weight, np.array([1, *input])))
        return label

    def train(self, training_inputs, training_labels, testing_inputs, testing_labels):
        """
        Train the perceptron
        Args:
            training_inputs: list of numpy arrays of training points.
            training_labels: arrays of expected output value for the corresponding point in training_inputs.
            testing_inputs: list of numpy arrays of testing points.
            testing_labels: arrays of expected output value for the corresponding point in testing_inputs.
        """
        self.weight = np.zeros(len(training_inputs[0]) + 1)
        accu_train = []
        accu_test = []
        for _ in range(self.max_epochs):
            for i in range(self.n_inputs):
                if self.forward(training_inputs[i]) * training_labels[i] <= 0:
                    self.weight += training_labels[i] * self.learning_rate * np.array([1, *training_inputs[i]])
            accu_train.append(self.accuracy(training_inputs, training_labels))
            accu_test.append(self.accuracy(testing_inputs, testing_labels))
        return accu_train, accu_test

    # calculate the accuracy of data X and labels y
    def accuracy(self, X, y):
        correct = 0
        for i in range(len(y)):
            if self.forward(X[i]) * y[i] > 0:
                correct += 1
        return correct / len(y)

    # split data X into positive set and negative set by the prediction of the perceptron
    def get_predict(self, X, y):
        pos = []
        neg = []
        for i in range(len(y)):
            if self.forward(X[i]) >= 0:
                pos.append(X[i])
            else:
                neg.append(X[i])
        return np.array(pos), np.array(neg)


# shuffle data X with its label y
def shuffle(X, y):
    randomize = np.arange(len(y))
    np.random.shuffle(randomize)
    return X[randomize], y[randomize]


# turn the raw data X1 X2 into the training set and testing set
def generate_data(X1, X2):
    X_train = np.concatenate((X1[:160], X2[:160]), axis=0)
    y_train = np.append(np.ones(160), -np.ones(160))
    X_train, y_train = shuffle(X_train, y_train)

    X_test = np.concatenate((X1[160:], X2[160:]), axis=0)
    y_test = np.append(np.ones(40), -np.ones(40))
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    mu1, mu2 = 2, 5
    sigma1, sigma2 = 2, 2
    X1 = np.random.normal(mu1, sigma1, (200, 2))
    X2 = np.random.normal(mu2, sigma2, (200, 2))
    X_train, y_train, X_test, y_test = generate_data(X1, X2)

    perceptron = Perceptron(320)
    train_accu, test_accu = perceptron.train(X_train, y_train, X_test, y_test)

    print('Final test accuracy: {}'.format(test_accu[-1]))