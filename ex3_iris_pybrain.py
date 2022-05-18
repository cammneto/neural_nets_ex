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

print(inputs[0:4])
