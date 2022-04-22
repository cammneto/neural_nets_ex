#!/bin/python3
import numpy as np
import neuralnets_functions as nnf

inputs  = np.array([[0,0],[0,1],[1,0],[1,1]])
outputs = np.array([[0],[1],[1],[0]])
weights0 = np.array([[-0.424,-0.740,-0.961],[0.358,-0.577,-0.469]])
weights1 = np.array([[-0.017],[-0.893],[0.148]])

epochs = 100
learning_rate = 0.3

#for epoch in epochs:

input_layer = inputs

sum_synapse0 = np.dot(input_layer, weights0)
#print('sum_synapse0','\n',sum_synapse0)
hidden_layer = sigmoid(sum_synapse0)
#print('hidden_layer','\n',hidden_layer)
sum_synapse1 = np.dot(hidden_layer, weights1)
#print('sum_synapse1','\n',sum_synapse1)
output_layer = sigmoid(sum_synapse1)
print('outpu_layer','\n',output_layer)
error_output_layer = outputs - output_layer
print('error_output_layer','\n',error_output_layer)
average = np.mean(abs(error_output_layer))
print('average','\n',average)


