#!/bin/python3
import numpy as np
import neuralnets_functions as nnf
from sklearn import datasets as dtst

iris = dtst.load_iris()

inputs  = iris.data[0:100]
outputs = iris.target[0:100]
outputs = outputs.reshape(-1,1)
weights0 = 2 * np.random.random((4, 5)) - 1
weights1 = 2 * np.random.random((5, 1)) - 1

epochs = 3000
learning_rate = 0.01
error = []
for epoch in range(epochs):
    input_layer = inputs
    sum_synapse0 = np.dot(input_layer, weights0)
    hidden_layer = nnf.sigmoid(sum_synapse0)
    sum_synapse1 = np.dot(hidden_layer, weights1)
    output_layer = nnf.sigmoid(sum_synapse1)
    error_output_layer = outputs - output_layer
    average = np.mean(abs(error_output_layer))
    if average < 10**(-2):
        break
    if epoch % 1000 == 0:
        print('Epoch: ' + str(epoch + 1) + ' Error: ' + str(average))
        error.append(average)
    derivative_output = nnf.sigmoid_derivative(output_layer)
    delta_output = error_output_layer * derivative_output
    delta_output_x_weight = delta_output.dot(weights1.T)
    delta_hidden_layer = delta_output_x_weight * nnf.sigmoid_derivative(hidden_layer)
    input_x_delta1 = (hidden_layer.T).dot(delta_output)
    weights1 = weights1 + (input_x_delta1 * learning_rate)
    input_x_delta0 = (input_layer.T).dot(delta_hidden_layer)
    weights0 = weights0 + (input_x_delta0*learning_rate)
print('Epoch: ' + str(epoch + 1) + ' Error: ' + str(average))
#print(inputs)
#for i in range(len(inputs)):
print(iris.target_names[int(round(nnf.sigmoid_output(50,weights0,weights1)))])

