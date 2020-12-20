
import cupy as cp
from .layers import Layer
from tqdm import tqdm

class NN(Layer):
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def forward(self, x):
        for i in self.layers:
            x = i(x)
        return x

    def backward(self, output_error, learning_rate = 0.001):
        for i in reversed(self.layers):
            output_error = i.backward(output_error, learning_rate)

    # set loss to use
    def use_loss(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in tqdm(range(epochs), desc = "training: "):
            for j in range(samples):
                # forward propagation
                input = x_train[j]

                output = self.forward(input)

                # compute loss (for display purpose only)
                err = self.loss(y_train[j], output)

                # backward propagation
                grad = self.loss_prime(y_train[j], output)  ### shape is (batch_size,1)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            # calculate average error on all samples
            err /= samples
            # print('epoch %d/%d   error=%f' % (i+1, epochs, err))