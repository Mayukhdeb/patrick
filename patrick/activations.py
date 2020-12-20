from .layers import ActivationLayer
from .utils import  __tanh__,  __tanh_prime__
from .utils import  __relu__, __relu_prime__
from .utils import  __leaky_relu__, __leaky_relu_prime__
from .utils import __sigmoid__, __sigmoid_prime__

class leaky_relu(ActivationLayer):
    def __init__(self):
        super().__init__(activation= __leaky_relu__, activation_prime = __leaky_relu_prime__)
       
class sigmoid(ActivationLayer):
    def __init__(self):
        super().__init__(activation= __sigmoid__, activation_prime = __sigmoid_prime__)
       
class tanh(ActivationLayer):
    def __init__(self):
        super().__init__(activation= __tanh__, activation_prime = __tanh_prime__)
       
class relu(ActivationLayer):
    def __init__(self):
        super().__init__(activation= __relu__, activation_prime = __relu_prime__)