### Introduction

This is an implementation of a  deep L-layer Neural Network. Here ,it is applied to detect images of cats. However, it's application in the real world range from computer vision, speech recognition, natural language processing, machine translation ,and bioinformatics.

## Pipeline
###1. Import Packages
First, import all the packages you'll need .
numpy is the main package for scientific computing with Python.
matplotlib is a library to plot graphs in Python.
np.random.seed(1) is used to keep all the random function calls consistent.

###2. Outline 
To build your neural network, several "helper functions have been implemented ." These helper functions will be used to build an L-layer neural network.

 Here's an outline of the steps in this assignment:

1.Initialize the parameters for an  𝐿 -layer neural network
2.Implement the forward propagation module 
3.Complete the LINEAR part of a layer's forward propagation step (resulting in  𝑍[𝑙] ).
4.The ACTIVATION function is provided for you (relu/sigmoid)
5.Combine the previous two steps into a new [LINEAR->ACTIVATION] forward function.
6.Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add a [LINEAR->SIGMOID] at the end (for the final layer  𝐿 ). This gives you a new L_model_forward function.
7.Compute the loss
8.Implement the backward propagation module (denoted in red in the figure below)
9.Complete the LINEAR part of a layer's backward propagation step
10.The gradient of the ACTIVATE function is provided for you(relu_backward/sigmoid_backward)
11.Combine the previous two steps into a new [LINEAR->ACTIVATION] backward function
12.Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new L_model_backward function
13.Finally, update the parameters

###3. Intialize parameters

Use random initialization for the weight matrices. Use np.random.randn(shape) * 0.01.
Use zeros initialization for the biases. Use np.zeros(shape).
You'll store  𝑛[𝑙] , the number of units in different layers, in a variable layer_dims

###4. Forward Propagation
####Linear Forward
![](linear_forward.png)