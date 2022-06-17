#!/bin/python3
import numpy as np
import neuralnets_functions as nnf
from sklearn import datasets
import pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.structure.modules import SigmoidLayer
from pybrain.structure.modules import SoftmaxLayer

iris = datasets.load_iris()
inputs = iris.data
outputs = iris.target


network = buildNetwork(4, 4, 4, 3, outclass = SoftmaxLayer, hiddenclass = SigmoidLayer, bias = False)

dataset = SupervisedDataSet(4, 3)
for i in range(len(inputs)):
  if outputs[i] == 0:
    output = (1, 0, 0, )
  elif outputs[i] == 1:
    output = (0, 1, 0,)
  else:
    output = (0, 0, 1, )
  dataset.addSample((inputs[i][0], inputs[i][1], inputs[i][2], inputs[i][3]), output)

#print(dataset['input'][:5])
optimizer = BackpropTrainer(module=network, dataset = dataset, learningrate = 0.01)
epochs = 3000
error = []
for epoch in range(epochs):
    error_average = optimizer.train()
    if error_average < 0.005:
        break
    if epoch % 100 == 0:
        print('Epoch: ' + str(epoch + 1) + ' Error: ' + str(error_average))
        error.append(error_average)
print('Epoch: ' + str(epoch + 1) + ' Error: ' + str(error_average))

import matplotlib.pyplot as plt
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.plot(error)
plt.show()

print(np.argmax(network.activate(inputs[101])))
