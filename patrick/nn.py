
import cupy as cp
from .layers import Layer
from tqdm import tqdm

class NN(Layer):
    def __init__(self):
        self.layers = []

    def forward(self, x):
        for i in self.layers:
            x = i(x)
        return x

    def backward(self, output_error, learning_rate = 0.001):
        for i in reversed(self.layers):
            output_error = i.backward(output_error, learning_rate)

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate, loss):
        # sample dimension first
        num_batches = len(x_train)

        # training loop
        for i in tqdm(range(epochs), desc = "training: "):
            for j in range(num_batches):
                # forward propagation
                batch = x_train[j]

                pred = self.forward(batch)

                # compute loss (for display purpose only)
                err = loss(pred = pred, label = y_train[j])

                # backward propagation
                grad = loss.prime(pred = pred , label = y_train[j])  
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            # calculate average error on all samples
            # err /= num_batches
            # print('epoch %d/%d   error=%f' % (i+1, epochs, err))