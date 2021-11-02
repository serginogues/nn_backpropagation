from network import NN
import matplotlib.pyplot as plt
from config import *
import numpy as np


def plot_all():
    nn = NN()
    nn.train()
    plt.plot(np.array(range(nn.epochs_run)), nn.cost)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")

    titl = "Alpha=" + str(LR) + ", Lambda = " + str(WD)

    configuration = ""
    if WEIGHT_INIT == 0:
        configuration += "Weights initialised with random uniform distribution"
    elif WEIGHT_INIT == 1:
        configuration += "All weights initialised with 0 values"
    elif WEIGHT_INIT == 2:
        configuration += "All weights initialised with random values between 0 and 1"
    print('')
    print("Plot configuration:", configuration)
    plt.title(titl)

    print('')
    print("Weights layer 1:", nn.synapse_0)
    print('')
    print("Weights layer 2:", nn.synapse_1)
    print('')
    print("Activation values hidden layer", nn.activation_hidden)

    plt.show()