import cupy as cp

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from patrick.nn import NN as nn
from patrick.losses import mse_loss
from patrick.activations import leaky_relu
from patrick.layers  import FCLayer as linear

"""
train test split
"""
data_bunch = load_digits()
x_train, y_train = cp.array(data_bunch.data), cp.array(data_bunch.target)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.1,shuffle = True)

"""
one hot encoding
"""
encoder = OneHotEncoder()
y_train = cp.array(encoder.fit_transform(y_train.get().reshape(-1,1)).toarray())

# print(x_train.shape, y_train.shape)

"""
split into batches
"""
batch_size = 7
x_train = x_train.reshape(x_train.shape[0]//batch_size, batch_size, x_train.shape[1]) /16.0
y_train = y_train.reshape(y_train.shape[0]//batch_size, batch_size, 10)

"""
model
"""
class model(nn):
    def __init__(self):
        self.layers =  [
                    linear(64,150),
                    leaky_relu(),
                    linear(150, 100),
                    leaky_relu(),
                    linear(100, 52),
                    leaky_relu(),
                    linear(52,10)
                ]
        
net = model()

"""
train
"""
net.fit(x_train, y_train, epochs=60, learning_rate=0.005, loss = mse_loss)

"""
test
"""
out = net(x_test/16.0).get()
out = np.array([np.argmax(i) for i in out]).astype(np.uint8)
labels = y_test.get().flatten()
print("Accuracy:", accuracy_score( labels, out)) 