import cupy as cp

# loss function and its derivative
def mse(y_true, y_pred):
    # print(y_true, y_pred)
    loss = cp.mean(cp.power(y_true-y_pred, 2))
    # print("losss", loss)
    return loss

def mse_prime(y_true, y_pred):
    return  2*(y_pred-y_true)/y_true.size