#!/bin/python3
import numpy as np
def step_function(sum):
    if (sum >= 1):
        return 1
    return 0

def calculate_output(instance,weights):
    s = instance.dot(weights)
    return step_function(s)

def npnorm(data):
    return data/np.linalg.norm(data)

def sigmoid(sum):
  return 1 / (1 + np.exp(-sum))

def sigmoid_derivative(sigmoid):
    return sigmoid*(1 - sigmoid)
