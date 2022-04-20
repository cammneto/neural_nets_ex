import numpy as np
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
outputs = np.array([0,1,1,1])
weights = np.array([0.0,0.0])
learning_rate = 0.1

def step_function(sum):
    if (sum >= 1):
        return 1
    return 0

def calculate_output(instance):
    s = instance.dot(weights)
    return step_function(s)

#calculate_output(np.array([[1,1]]))

def train():
    total_error = 1
    while (total_error != 0):
        total_error = 0
        for i in range(len(outputs)):
            prediction = calculate_output(inputs[i])
            error = abs(outputs[i] - prediction)
            total_error += error
            if error > 0:
                for j in range(len(weights)):
                    weights[j] = weights[j] + (learning_rate*inputs[i][j]*error)
                    print('Weight updated: ' + str(weights[j]))
        print('Total error: ' + str(total_error))

train()