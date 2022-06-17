import pybrain

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.structure.modules import SigmoidLayer

network = buildNetwork(2,3,1,outclass = SigmoidLayer, bias = False)

dataset = SupervisedDataSet(2,1)
dataset.addSample((0,0),(0,))
dataset.addSample((0,1),(1,))
dataset.addSample((1,0),(1,))
dataset.addSample((1,1),(0,))

optimizer = BackpropTrainer(module=network, dataset = dataset, learningrate = 0.3)
epochs = 5000
error = []
for epoch in range(epochs):
    error_average = optimizer.train()
    if epoch % 1000 == 0:
        print('Epoch: ' + str(epoch + 1) + ' Error: ' + str(error_average))
        error.append(error_average)
print('Epoch: ' + str(epoch + 1) + ' Error: ' + str(error_average))

import matplotlib.pyplot as plt
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.plot(error)
plt.show()