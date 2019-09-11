#Importing Library
import numpy as np
from onn.OnlineNeuralNetwork import ONN
print("np.asarray([[0.8, 0.5]])",np.asarray([[0.8, 0.5]]).shape)

print("np.asarray([[0.8, 0.5]])",np.asarray([[0.8, 0.5]])[0].shape)
print(np.asarray([[0,1]]).shape)
#Starting a neural network with feature size of 2, hidden layers expansible until 5, number of neuron per hidden layer = 10 #and two classes.
onn_network = ONN(features_size=2, max_num_hidden_layers=5, qtd_neuron_per_hidden_layer=10, n_classes=2)

#Do a partial training
onn_network.partial_fit(np.asarray([[0.1, 0.2]]), np.asarray([0,1]))
onn_network.partial_fit(np.asarray([[0.8, 0.5]]), np.asarray([1,0]))


#Predict classes
onn_network.partial_fit(np.asarray([[0.8, 0.5]]), np.asarray([1]))
predictions = onn_network.predict(np.asarray([[0.1, 0.2], [0.8, 0.5]]))

print(predictions)
#-- array([1, 0])