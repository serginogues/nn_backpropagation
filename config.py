# Do not change
INPUT_DIM = 8
HIDDEN_DIM = 3
OUTPUT_DIM = 8
EPOCHS = 50000

# Hyperparams to tune
LR = 0.01  # learning rate
WD = 0.001  # weight decay
WEIGHT_INIT = 0
"""
WEIGHT_INIT: Weight initialization approach
0 - uniform distribution
1 - zeros
2 - random between 0 and 1
"""
