import cupy as cp
from losses import mse, mse_prime
from activations import  leaky_relu
from layers import FCLayer
from nn import NN

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

"""
train test split
"""
iris = datasets.load_iris()
x_train, y_train = cp.array(iris.data), cp.array(iris.target)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.1,shuffle = True)

"""
one hot encoding
"""
encoder = OneHotEncoder()
y_train = cp.array(encoder.fit_transform(y_train.get().reshape(-1,1)).toarray())

"""
split into batches
"""
batch_size = 3
x_train = x_train.reshape(x_train.shape[0]//batch_size, batch_size, x_train.shape[1])
y_train = y_train.reshape(y_train.shape[0]//batch_size, batch_size, 3)

"""
model
"""
class model(NN):
    def __init__(self):
        self.layers =  [
                    FCLayer(4,12),
                    leaky_relu(),
                    FCLayer(12, 10),
                    leaky_relu(),
                    FCLayer(10,3) ## could be one hot encoding, but for now this will do
                ]
        self.loss = mse
        self.loss_prime = mse_prime
        
net = model()

"""
train
"""
net.fit(x_train, y_train, epochs=100, learning_rate=0.001)

"""
test
"""
out = net(x_test).get()
out = np.array([np.argmax(i) for i in out]).astype(np.uint8)
labels = y_test.get().flatten()
print("Accuracy:", accuracy_score( labels, out))