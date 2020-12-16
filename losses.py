import cupy as cp

# loss function and its derivative
def mse(y_true, y_pred):
    return cp.mean(cp.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size
