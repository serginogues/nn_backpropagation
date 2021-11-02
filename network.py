"""
Shallow neural network architecture:
    - Input layer of size 8 + 1 (bias node returning 1)
    - 1 hidden layer of size 3 + 1 (bias node returning 1)
    - Output layer of size 8
Input and output is the same one-hot binary vector.
Example: [0,1,0,0,0,0,0,0] -> nn -> [0,1,0,0,0,0,0,0]

Thus, only 8 learning samples.

Use Sigmoid as ctivation function because we are doing softmax regression.
"""
import numpy as np

from config import *


class NN:
    """
    Shallow neural network
    """

    def __init__(self, weight_config=WEIGHT_INIT, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, lr=LR, w_d=WD, epochs=EPOCHS):
        """
        :param x: learning samples, a 2D array of shape (8,8)
        :param y: desired output, a 2D array of shape (8,8)
        :param input_dim: input dimension
        :param hidden_dim: hidden layer dimension
        :param output_dim: output dimension
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dataset = self.generate_samples()
        self.y = self.generate_samples()
        self.lr = lr
        self.weight_decay = w_d
        self.epochs = epochs

        self.cost = []
        self.epochs_run = 0
        self.activation_hidden = np.zeros((3,))

        if weight_config == 0:
            self.synapse_0 = np.random.uniform(-.05, .05, (self.hidden_dim, self.input_dim + 1))
            self.synapse_1 = np.random.uniform(-.05, .05, (self.output_dim, self.hidden_dim + 1))
        elif weight_config == 1:
            self.synapse_0 = np.zeros((self.hidden_dim, self.input_dim + 1))
            self.synapse_1 = np.zeros((self.output_dim, self.hidden_dim + 1))
        elif weight_config == 2:
            self.synapse_0 = np.random.random((self.hidden_dim, self.input_dim + 1))
            self.synapse_1 = np.random.random((self.output_dim, self.hidden_dim + 1))

        self.gradients_0 = np.zeros(self.synapse_0.shape)
        self.gradients_1 = np.zeros(self.synapse_1.shape)

    def train(self):
        train = True
        epoch = 0
        while train and epoch < EPOCHS:
            epoch = epoch+1
            print("epoch:", epoch)
            self.fit()
            if epoch % 10 == 0 and self.test():
                train = False
        self.epochs_run = epoch
        self.test(True)

    def test(self, print_=False):
        n_correct = 0
        CONVERGED = False
        if print_:
            print('')

        for i, sample in enumerate(self.dataset):
            predicted_y, a_hidden_layer = self.forward(sample)
            if np.argmax(sample) == np.argmax(predicted_y):
                n_correct += 1
            if print_:
                v = np.zeros(sample.shape)
                v[np.argmax(predicted_y)] = 1
                print(str(i) + ":")
                print("label:")
                print(sample)
                print("predicted:")
                print(v)
        if print_:
            print("Number of correct classes:", n_correct)

        if n_correct == 8:
            CONVERGED = True

        return CONVERGED

    def fit(self):
        # one gradient per weight
        self.gradients_0 = np.zeros(self.synapse_0.shape)
        self.gradients_1 = np.zeros(self.synapse_1.shape)
        cost = 0

        for sample in self.dataset:
            # get rid of the bias parameter
            desired_y = sample
            # sample (i.e.) = [0,1,0,0,0,0,0,0]

            # forward pass
            predicted_y, a_hidden_layer = self.forward(sample)

            # print the cost
            cost = cost + 0.5 * (np.linalg.norm(predicted_y - desired_y) ** 2)

            # compute the delta errors by doing a backward pass
            delta_output_layer, delta_hidden_layer = self.compute_deltas(predicted_y, desired_y, a_hidden_layer)

            # partial derivatives for each weight
            self.update_gradients(delta_output_layer, a_hidden_layer, delta_hidden_layer, sample)

        print(cost / self.input_dim)
        self.cost.append(cost / self.input_dim)
        self.update_weights()

    def activation(self, input, layer):
        """
        Neuron activation is calculated as the weighted sum of the inputs (like linear regression) + sigmoid
        We assume the bias to be the first row of weights connecting an input node that returns 1 with the three hidden nodes
        """
        if layer == 1:
            activation = np.zeros((3,))
            weights = self.synapse_0

        else:
            activation = np.zeros((8,))
            weights = self.synapse_1

        # sum + bias
        z = np.dot(weights[:, 1:], input) + weights[:, 0]

        # sigmoid transfer
        activation = self.sigmoid(z)

        return activation

    def forward(self, x):
        a_hidden_layer = self.activation(x, 1)
        predicted_y = self.activation(a_hidden_layer, 2)

        self.activation_hidden = a_hidden_layer

        return predicted_y, a_hidden_layer

    def compute_deltas(self, predicted_y, desired_y, a_hidden_layer):
        """
        Compute the delta error vectors for the output and hidden layer
        :param predicted_y: same as input since we just want to copy
        :param desired_y: output of neural network
        :param a_hidden_layer: activation values for each of the 3 nodes in hidden layer
        """
        # compute the error delta(3) = delta_output_layer
        delta_output_layer = -1 * (desired_y - predicted_y) * self.sigmoid_derivative(predicted_y)

        # compute delta(2)
        delta_hidden_layer = np.dot(self.synapse_1.T[1:], delta_output_layer) * self.sigmoid_derivative(a_hidden_layer)

        return delta_output_layer, delta_hidden_layer

    def update_gradients(self, delta_output_layer, a_hidden_layer, delta_hidden_layer, sample):
        """ Update the partial derivatives """

        # synapse 1
        for i in range(len(delta_output_layer)):
            self.gradients_1[i, 0] = self.gradients_1[i, 0] + delta_output_layer[i]
            for j in range(len(a_hidden_layer)):
                self.gradients_1[i, j + 1] = self.gradients_1[i, j + 1] + a_hidden_layer[j] * delta_output_layer[i]

        # synapse 0
        for i in range(len(delta_hidden_layer)):
            self.gradients_0[i, 0] = self.gradients_0[i, 0] + delta_hidden_layer[i]
            for j in range(len(sample)):
                self.gradients_0[i, j + 1] = self.gradients_0[i, j + 1] + sample[j] * delta_hidden_layer[i]

    def update_weights(self):
        # compute cost gradient dJ/dtheta
        cost_gradient_synapse1 = np.zeros(self.synapse_1.shape)
        cost_gradient_synapse0 = np.zeros(self.synapse_0.shape)

        cost_gradient_synapse1[:, 1:] = (self.gradients_1[:, 1:] + (self.weight_decay * self.synapse_1[:, 1:])) / self.input_dim
        cost_gradient_synapse0[:, 1:] = (self.gradients_0[:, 1:] + (self.weight_decay * self.synapse_0[:, 1:])) / self.input_dim

        cost_gradient_synapse1[:, 0] = self.gradients_1[:, 0] / self.input_dim
        cost_gradient_synapse0[:, 0] = self.gradients_0[:, 0] / self.input_dim

        # update weights with gradient descent
        self.synapse_0 = self.synapse_0 - self.lr * cost_gradient_synapse0
        self.synapse_1 = self.synapse_1 - self.lr * cost_gradient_synapse1

    @staticmethod
    def sigmoid_derivative(a):
        return a * (1 - a)

    @staticmethod
    def sigmoid(z):
        """
        Activation function
        """
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def generate_samples():
        """
        Generate one-hot vectors foreach of the 8 training samples
        """
        a = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        b = np.zeros((a.size, a.size))
        b[a, a] = 1
        return b