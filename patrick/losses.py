import cupy as cp
from .layers import Layer

# loss function and its derivative

def mse(pred, label):
    # print(y_true, y_pred)
    # print(pred,label)
    loss = cp.mean(cp.power(pred-label, 2))
    # print("mse", loss)
    # print("losss", loss)
    return loss

def mse_prime(pred,label):
    return  2*(pred-label)/label.size

class loss_function(Layer):
    def __init__(self, loss_func, loss_func_prime):
        self.loss_func = loss_func
        self.loss_func_prime = loss_func_prime

    def forward(self, pred,label):
        return self.loss_func(pred,label)

    def prime(self, pred, label):
        return self.loss_func_prime(pred, label)

    def __call__(self, pred, label):
        return self.forward(pred, label)



mse_loss = loss_function(loss_func = mse, loss_func_prime= mse_prime)
