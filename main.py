"""
Advanced Concepts in Machine Learning
Assignment 1: Backpropagation from scratch

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


class NN:
    """
    Shallow neural network
    """

    def __init__(self, input_dim=8, hidden_dim=3, output_dim=8, lr=0.01, w_d=0.001):
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
        self.input = self.generate_samples()
        self.y = self.generate_samples()
        self.output = np.zeros(self.y.shape)
        self.lr = lr
        self.weight_decay = w_d
        # nn weights with 1 bias node per layer (except output layer)
        self.synapse_0 = np.random.random((self.input_dim+1, self.hidden_dim))
        self.synapse_1 = np.random.random((self.hidden_dim+1, self.output_dim))

    def train(self):
        for _ in range(10000):
            self.fit()

    def test(self):
        n_correct = 0
        n_samples = 8
        for sample in self.input:
            predicted_y, a_hidden_layer, z_hidden, z_output = self.forward(sample)
            if (predicted_y == sample).all():
                n_correct += 1

        print("n_correct = " + str(n_correct))

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
        z = np.dot(input, weights[1:]) + weights[0]

        # sigmoid transfer
        activation = self.sigmoid(z)

        return activation, z

    def forward(self, x):
        a_hidden_layer, z_hidden = self.activation(x, 1)
        predicted_y, z_output = self.activation(a_hidden_layer, 2)

        return predicted_y, a_hidden_layer, z_hidden, z_output

    def compute_deltas(self, predicted_y, desired_y, a_hidden_layer):
        """
        Compute the delta error vectors for the output and hidden layer
        :param predicted_y: same as input since we just want to copy
        :param desired_y: output of neural network
        :param a_hidden_layer: activation values for each of the 3 nodes in hidden layer
        """
        # compute the error delta(3) = delta_output_layer
        delta_output_layer = -1 * self.sigmoid_derivative(predicted_y) * (desired_y - predicted_y)

        # compute delta(2)
        delta_hidden_layer = np.dot(delta_output_layer, self.synapse_1.T[:, 1:]) * self.sigmoid_derivative(a_hidden_layer)

        return delta_output_layer, delta_hidden_layer

    def weight_derivative(self, delta, activation):
        """ Compute desired partial derivatives """
        d_weights = np.zeros((len(activation), len(delta)))
        for j in range(len(activation)):
            for i in range(len(delta)):
                d_weights[j, i] += activation[j] * delta[i]

        return d_weights

    def bias_derivative(self, delta):
        d_bias = np.zeros((1, len(delta)))
        for i in range(len(delta)):
            d_bias[0, i] += delta[i]

        return d_bias

    def update_weights(self, d_weights_1, d_bias_1, d_bias_0, d_weights_0):
        # compute cost gradient dJ/dtheta
        cost_gradient_synapse1 = np.zeros(self.synapse_1.shape)
        cost_gradient_synapse0 = np.zeros(self.synapse_0.shape)
        m_1 = 1 / self.input_dim

        cost_gradient_synapse1[1:, :] = (m_1 * d_weights_1) + (self.weight_decay * self.synapse_1[1:, :])
        cost_gradient_synapse0[1:, :] = (m_1 * d_weights_0) + (self.weight_decay * self.synapse_0[1:, :])

        cost_gradient_synapse1[0, :] = m_1 * d_bias_1
        cost_gradient_synapse0[0, :] = m_1 * d_bias_0

        # update weights with gradient descent
        self.synapse_0 -= self.lr * cost_gradient_synapse0
        self.synapse_1 -= self.lr * cost_gradient_synapse1

    def fit(self):
        # one gradient per weight
        d_weights_0 = np.zeros((self.input_dim, self.hidden_dim))
        d_bias_0 = np.zeros((1, self.hidden_dim))
        d_weights_1 = np.zeros((self.hidden_dim, self.output_dim))
        d_bias_1 = np.zeros((1, self.output_dim))
        average_cost = 0

        for sample in self.input:
            # get rid of the bias parameter
            desired_y = sample
            # sample (i.e.) = [0,1,0,0,0,0,0,0]

            # forward pass
            predicted_y, a_hidden_layer, z_hidden, z_output = self.forward(sample)

            # print the cost
            cost = 0.5 * (np.abs(predicted_y-desired_y)**2)
            average_cost += np.mean(cost)

            # compute the delta errors by doing a backward pass
            delta_output_layer, delta_hidden_layer = self.compute_deltas(predicted_y, desired_y, a_hidden_layer)

            # partial derivatives for each weight
            d_weights_1 += self.weight_derivative(delta_output_layer, a_hidden_layer)
            d_bias_1 += self.bias_derivative(delta_output_layer)
            d_weights_0 += self.weight_derivative(delta_hidden_layer, sample)
            d_bias_0 += self.bias_derivative(delta_hidden_layer)

        print(average_cost/self.input_dim)
        self.update_weights(d_weights_1, d_bias_1, d_bias_0, d_weights_0)

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


if __name__ == '__main__':
    nn = NN()
    # nn.test()
    nn.train()
    nn.test()
