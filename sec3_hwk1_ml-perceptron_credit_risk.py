#!/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neuralnets_functions as nnf
from sklearn.preprocessing import MinMaxScaler as mms

df1 = pd.read_csv('credit_data.csv')
df1 = df1.dropna()

inputs = df1.iloc[:,1:4].values
scaler = mms()
inputs = scaler.fit_transform(inputs)
#inputs = nnf.npnorm(inputs)
print(inputs)
outputs = np.array([df1['c#default']])
outputs = outputs.T
weights0 = 2 * np.random.random((3, 10)) - 1
weights1 = 2 * np.random.random((10, 1)) - 1

epochs = 100000
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
    if average < 8*10**(-2):
        break
    if epoch % 10 == 0:
        #print('Epoch: ' + str(epoch + 1) + ' Error: ' + str(average))
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
print('output_layer','\n', np.rint(output_layer))

plt.xlabel('Epochs')
plt.ylabel('Error')
plt.plot(error)
plt.show()

def calculate_output(instance):
  hidden_layer = nnf.sigmoid(np.dot(instance, weights0))
  output_layer = nnf.sigmoid(np.dot(hidden_layer, weights1))
  return output_layer[0]

#print(round(calculate_output(inputs[4])))
