#!/bin/python3
import numpy as np
import neuralnets_functions as nnf

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
outputs = np.array([0,1,1,1])
weights = np.array([0.0,0.0])
learning_rate = 0.1

def train():
    total_error = 1
    while (total_error != 0):
        total_error = 0
        for i in range(len(outputs)):
            prediction = nnf.calculate_output(inputs[i],weights)
            error = abs(outputs[i] - prediction)
            total_error += error
            if error > 0:
                for j in range(len(weights)):
                    weights[j] = weights[j] + (learning_rate*inputs[i][j]*error)
                    print('Weight updated: ' + str(weights[j]))
        print('Total error: ' + str(total_error))

train()