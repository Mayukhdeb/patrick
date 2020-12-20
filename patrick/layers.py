import cupy as cp 

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_error, learning_rate):
        raise NotImplementedError

    def __call__(self, input):
        return self.forward(input)

# inherit from base class Layer
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there are no "learnable" parameters.
    def backward(self, output_error, learning_rate):
        # print(output_error.shape)
        return self.activation_prime(self.input) * output_error

    def __call__(self, input):
        return self.forward(input)

        
class FCLayer(Layer):

    def __init__(self, input_size, output_size):
        self.weights = cp.random.rand(input_size, output_size) -0.5
        self.bias = cp.random.rand(1, output_size) -0.5

    # returns output for a given input
    def forward(self, input_data):
        self.input = input_data
        self.output = cp.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_grad, learning_rate):

        """
        Z = W(x) + B
        """

        # print("dloss", output_grad.T.shape, self.input.shape)

        weights_grad = cp.dot( output_grad.T,  self.input )
        biases_grad = cp.mean(output_grad , axis = 0)
        input_grad = cp.dot(output_grad, self.weights.T)

        # print("bb", biases_grad.shape)

        # print("weights grad shape:", weights_grad.shape)
        # print("adder shape:", (learning_rate*weights_grad.T).shape, "weigts shape", self.weights.shape)

        self.weights -=  (learning_rate*weights_grad.T)
        self.bias -= learning_rate*biases_grad
        
        return input_grad

    def __call__(self, input):
        return self.forward(input)


class LossLayer(Layer):
    def __init__(self, loss_func, loss_func_prime):
        self.loss_func = loss_func
        self.loss_func_prime = loss_func_prime

    def forward(self, pred,label):
        return self.loss_func(pred,label)

    def prime(self, pred, label):
        return self.loss_func_prime(pred, label)

    def __call__(self, pred, label):
        return self.forward(pred, label)