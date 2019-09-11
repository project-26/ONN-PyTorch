import numpy as np
from sklearn.model_selection import train_test_split
import os 
from scipy.io import loadmat
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import backend as K

def load(dataset):
    assert dataset in ['susy']
    #'higgs', 'cd0','cd2','cd3','cd4','cd5','cd6','cd7','syn8',
    #data = loadmat('../../data/' + dataset)
    x_data=np.load('../data/' + dataset+"/x_train.npy")
    y_data = np.load('../data/' + dataset + "/y_train.npy")



    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

    nb_classes = 2
    #print(len(x_train[0]))
    print(x_test.shape)
    print(y_test.shape)
    print(x_train.shape)
    print(y_train.shape)
    #print(y_train[0])
    #print(x_train)
    return (x_train, y_train,x_test, y_test, nb_classes)


