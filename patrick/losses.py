import cupy as cp
from .layers import LossLayer
from .utils import __mse__, __mse_prime__

mse_loss = LossLayer(loss_func = __mse__ , loss_func_prime= __mse_prime__)