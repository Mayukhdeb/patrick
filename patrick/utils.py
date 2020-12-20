import cupy as cp 

def __tanh__(x):
    return cp.tanh(x)

def __tanh_prime__(x):
    return 1-cp.tanh(x)**2

def __relu__(m):
    m[m < 0] = 0
    return m

def __relu_prime__(x):
    x[x>0] = 1
    x[x<=0] = 0
    return x

def __leaky_relu__(z, alpha = 0.2):
    pair = cp.array([alpha*z,z])
    return cp.max(pair, axis = 0)


def __leaky_relu_prime__(x, alpha = 0.2):
    x[x>0] = 1
    x[x<=0] = alpha
    return x

def __sigmoid__(x):
    x = 1/(1 + cp.exp(-x))
    return x

def __sigmoid_prime__(x):
    x =  (1/(1 + cp.exp(-x)))*(1- 1/(1 + cp.exp(-x)))
    return x

def __mse__(pred, label):
 
    loss = cp.mean(cp.power(pred-label, 2))

    return loss

def __mse_prime__(pred,label):
    return  2*(pred-label)/label.size