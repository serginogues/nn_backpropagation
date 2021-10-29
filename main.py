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

    def __init__(self, input_dim=8, hidden_dim=3, output_dim=8, lr=0.01):
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

        # nn weights with 1 bias node per layer (except output layer)
        self.synapse_0 = np.random.random((self.input_dim+1, self.hidden_dim))
        self.synapse_1 = np.random.random((self.hidden_dim+1, self.output_dim))

    def train(self):
        for _ in range(20000):
            self.fit()

    def test(self):
        n_correct = 0
        n_samples = 8
        for sample in self.input:
            predicted_y, a_hidden_layer = self.forward_pass(sample)
            if predicted_y.all() == sample.all():
                n_correct += 1

        print("n_correct = " + str(n_correct))

    def forward_pass(self, x):

        # first layer pass
        a_hidden_layer = self.sigmoid(np.dot(x, self.synapse_0))

        # insert bias node
        a_hidden_layer = np.insert(a_hidden_layer,0,1, axis=0)

        # second layer pass
        predicted_y = self.sigmoid(np.dot(a_hidden_layer, self.synapse_1))

        return predicted_y, a_hidden_layer

    def fit(self):
        # one gradient per weight
        synapse_0_gradients = np.zeros((self.input_dim+1, self.hidden_dim))
        synapse_1_gradients = np.zeros((self.hidden_dim+1, self.output_dim))
        average_cost = np.zeros((8, 8))

        for sample in self.input:
            # get rid of the bias parameter
            desired_y = sample[1:]
            # sample (i.e.) = [0,1,0,0,0,0,0,0]

            # forward pass
            predicted_y, a_hidden_layer = self.forward_pass(sample)

            cost = 0.5 * np.abs(predicted_y-desired_y)**2
            average_cost += cost

            # compute the error delta(3) = delta_output_layer
            delta_output_layer = self.sigmoid_derivative(predicted_y) * (predicted_y - desired_y)

            # compute delta(2) = delta_hidden_layer Remember: we ignore the weights of the bias since there is no
            # delta for the bias node - self.synapse_1.T[0,1:] = array([ theta1, theta2, theta3]) and theta0 is
            # ignored because comes from the bias. These 3 are the weights connecting output node 0 with the 3 nodes
            # of the hidden layer respectively - delta_output_layer.shape = (8,) meaning eight values for each of the
            # output nodes
            #ToDo: Ibelieve there is a bug here when computing delta_hidden_layer values (see slide 85 from week 1)
            delta_hidden_layer = np.zeros((3,))
            for i in range(delta_output_layer.size):
                delta_hidden_layer += self.synapse_1.T[i, 1:] * delta_output_layer[i]
            delta_hidden_layer = self.sigmoid_derivative(a_hidden_layer[1:]) * delta_hidden_layer

            # update gradients
            for j in range(self.hidden_dim):
                for i in range(self.output_dim):
                    synapse_1_gradients[j, i] += a_hidden_layer[j] * delta_output_layer[i]

            for j in range(self.input_dim):
                for i in range(self.hidden_dim):
                    synapse_0_gradients[j, i] += sample[j] * delta_hidden_layer[i]

        print(np.mean(average_cost, 1))

        # compute cost gradient dJ/dtheta
        decay_weights = 10  # try 0, 1, 10, 1000, 10000, 1000000
        cost_gradient_synapse1 = np.zeros(self.synapse_1.shape)
        cost_gradient_synapse0 = np.zeros(self.synapse_0.shape)
        cost_gradient_synapse1[1:, :] = (1 / self.input_dim) * (synapse_1_gradients[1:] + decay_weights * self.synapse_1[1:])
        cost_gradient_synapse0[1:, :] = (1 / self.input_dim) * (synapse_0_gradients[1:] + decay_weights * self.synapse_0[1:])

        cost_gradient_synapse1[0, :] = (1 / self.input_dim) * synapse_1_gradients[0]
        cost_gradient_synapse0[0, :] = (1 / self.input_dim) * synapse_0_gradients[0]

        # update weights with gradient descent
        self.synapse_0 = self.synapse_0 - self.lr * cost_gradient_synapse0
        self.synapse_1 = self.synapse_1 - self.lr * cost_gradient_synapse1

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def sigmoid(self, z):
        """
        Activation function
        """
        return 1 / (1 + np.exp(-z))

    def generate_samples(self):
        """
        Generate one-hot vectors foreach of the 8 training samples
        """
        a = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        b = np.zeros((a.size, a.size+1))
        b[a, a+1] = 1
        b[:,0] = 1
        # c = np.array([[b[i], b[i]] for i in range(8)])
        return b


if __name__ == '__main__':
    nn = NN()
    # nn.test()
    nn.train()
    nn.test()
