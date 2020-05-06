# MorphNet
Using Google's MorphNet to optimize a Deep Neural Network

# Problem
Problem is to optimize the Neural Network for Inverted Pendulum Dataset with respect to Computation (FLOPS) and Model Size.

# Steps
. Run inverted_pendulum_data_generation.py to save the data.
. Run flop_regularizer.py to generate optimized Neural Network with respect to Computation (FLOPS)
. Run modelsize_regularizer.py to generate optimized Neural Network with respect to Model Size
. Finally inorder to test the optimized neural network run text_NN.py

# Tweaking hyperparameters of MorphNet algorithm
. Under the flopregularizer class and modelsizeregularizer class in flop_regularizer.py and modelsize_regularizer.py file, regularization strength defines the strength by which neurons/synapses should be pruned. 
. Max the strength, more pruned is the network. Adjust it to obtain max accuracy.

