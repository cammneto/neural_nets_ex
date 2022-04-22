#!/bin/python3
import numpy as np
import neuralnets_functions as nnf

raw_inputs = np.array([[18,2], [20,3], [21, 4],[35,15], [36,16], [38, 18]])
outputs = np.array([0,0,0,1,1,1])
test_inputs = np.array([[17,5], [25,8],[45,10], [31,20]])
weights = np.array([0.0,0.0])
learning_rate = 0.1
inputs = nnf.npnorm(raw_inputs)

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

test_inputs = nnf.npnorm(test_inputs)

for i in range(len(test_inputs)):
    raise_sal = nnf.calculate_output(test_inputs[i],weights)
    print(raise_sal)
