import os, sys, getopt 
import yaml
#import cPickle
import pickle as cPickle

import numpy as np

import keras
import keras.callbacks
from keras.datasets import mnist
#from keras.utils.visualize_util import plot
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam, RMSprop

from keras.callbacks import CSVLogger
from data import load

def build_data_dict(in_name, out_name, in_data, out_data):
    in_dict = dict()
    in_dict[in_name] = in_data
    
    out_dict = dict((k, out_data) for k in out_name)
    return (in_dict, out_dict)

def build_loss_weight(config):
    if config['hedge'] == False:
        w = [1.]
    elif config['loss_weight'] == 'ave':
        w = [1./ config['n_layers']]* config['n_layers']
    return w
def main(arg, idx=0):
    print(arg)
    #arg = np.array(['-c', 'ml16.yaml'])
    print("after",arg)
    config = {'learning_rate': 1e-3,
              'optim': 'Adam',
              'batch_size': 1,
              'nb_epoch': 50,
              'n_layers': 3,
              'hidden_num': 100,
              'activation': 'relu',
              'loss_weight': 'ave',
              'adaptive_weight': False,
              'data': 'mnist',
              'hedge': False,
              'Highway': False,
              'momentum': 0.,
              'nesterov': False,
              'log': 'mnist_hedge.log'}

    configfile = ''
    helpstring = 'main.py -c <config YAML file>'
    try:
        opts, args = getopt.getopt(arg, "hc:", ["config"])
    except getopt.GetoptError:
        print(helpstring)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print (helpstring)
            yamlstring = yaml.dump(config,default_flow_style=False,explicit_start=True)
            print("YAML configuration file format:")
            print("")
            print("%YAML 1.2")
            print(yamlstring)
            sys.exit()

        elif opt in ('-c', '--config'):
            configfile = arg

        print("Config file is %s" % configfile)

    if os.path.exists(configfile):
        f = open(configfile)
        user_config = yaml.load(f.read())
        config.update(user_config)
    
    print("Printing configuration:")
    for key,value in config.items():
        print("  ",key,": ",value)

    (X_train, Y_train, X_test, Y_test, nb_classes) = load(config['data'])
    import numpy as np
    from onn.OnlineNeuralNetwork import ONN

    # Starting a neural network with feature size of 2, hidden layers expansible until 5, number of neuron per hidden layer = 10 #and two classes.
    onn_network = ONN(features_size=config['input_size'], max_num_hidden_layers=config['n_layers'], qtd_neuron_per_hidden_layer=100, n_classes=2,LOGDIR = '../logs/'+config['log'])
    #ONN(features_size=config['batch_size'], max_num_hidden_layers=config['n_layers'], qtd_neuron_per_hidden_layer=100,
        #n_classes=2)

    # Do a partial training
    for i in range(len(X_train)):  #len(X_train) 2000
        #print(np.asarray([X_train[i]]).shape)
        #print(np.asarray([Y_train[i]]).shape)
        onn_network.partial_fit(np.asarray([X_train[i]]), np.asarray([Y_train[i][0]]))
    #onn_network.partial_fit(np.asarray([[0.8, 0.5]]), np.asarray([1]))

    for i in range(len(X_test)):  #2000 #len(X_test)
        predictions = onn_network.predict(np.asarray([X_test[i]]))
        #predictions.
        #print("actual {}, predictions: {}".format(Y_test[i][0],predictions))
    #onn_network.summary()
    
    #plot(model, to_file = 'model.png')
    

if __name__ == '__main__':
    #for i in range(5):
    my_callback = main(sys.argv[1:], 0)
