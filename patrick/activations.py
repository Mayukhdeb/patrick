from .layers import ActivationLayer
from .utils import  __tanh__,  __tanh_prime__
from .utils import  __relu__, __relu_prime__
from .utils import  __leaky_relu__, __leaky_relu_prime__
from .utils import __sigmoid__, __sigmoid_prime__

tanh = ActivationLayer(__tanh__, __tanh_prime__)
relu = ActivationLayer(__relu__, __relu_prime__)
# leaky_relu = ActivationLayer(__leaky_relu__, __leaky_relu_prime__)
sigmoid = ActivationLayer(__sigmoid__, __sigmoid_prime__)

class leaky_relu(ActivationLayer):
    def __init__(self):
        super().__init__(activation= __leaky_relu__, activation_prime = __leaky_relu_prime__)
       