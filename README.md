# MorphNet
Using Google's MorphNet to optimize a Deep Neural Network

# Problem Statement
* Given the position of cart x, angle of rotation θ, velocity with respec to x, velocity with respect to θ, the problem is to predict the amount of external force (F) required to balance the inverted pendulum without falling down.
* To optimize the built Deep Neural Network with respect to Computation (FLOPS) and Model Size.


# Steps
* Run inverted_pendulum_data_generation.py to save the data.
* Run flop_regularizer.py to generate optimized Neural Network with respect to Computation (FLOPS)
* Run modelsize_regularizer.py to generate optimized Neural Network with respect to Model Size
* Finally inorder to test the optimized neural network run text_NN.py

# Tweaking hyperparameters of MorphNet algorithm
* Under the flopregularizer class and modelsizeregularizer class in flop_regularizer.py and modelsize_regularizer.py file, regularization strength defines the strength by which neurons/synapses should be pruned. 
* Max the strength, more pruned is the network. Adjust it to obtain max accuracy.

