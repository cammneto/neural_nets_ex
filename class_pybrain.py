import pybrain

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import SigmoidLayer, LinearLayer, BiasUnit
from pybrain.structure import FullConnection

network = FeedForwardNetwork()
input_layer = LinearLayer(2)
hidden_layer = SigmoidLayer(3)
output_layer = SigmoidLayer(1)
bias0 = BiasUnit()
bias1 = BiasUnit()

network.addModule(input_layer)
network.addModule(hidden_layer)
network.addModule(output_layer)
network.addModule(bias0)
network.addModule(bias1)

input_to_hidden = FullConnection(input_layer, hidden_layer)
hidden_to_output= FullConnection(hidden_layer, output_layer)
bias_hidden = FullConnection(bias0, hidden_layer)
bias_output = FullConnection(bias1, output_layer)

network.sortModules()
print(input_to_hidden.params)