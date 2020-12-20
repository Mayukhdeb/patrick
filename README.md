# patrick
spongebob's best friend

<p align="center">
<img src = "images/patrick.png" width = "20%">
</p>

Tiny neural net library written from scratch with [cupy](https://cupy.dev/) backend. Still under construction. 

```
from patrick.nn import NN as nn
from patrick.losses import mse_loss
from patrick.activations import leaky_relu
from patrick.layers  import FCLayer as linear

"""
Preprocess your data here 
shapes for x_train, y_train should be (num_batches, batch_size, input_size), (num_batches, batch_size, output_size)
"""

class Model(nn):
    def __init__(self):
        self.layers =  [
                    linear(64,150),  ## input size is 64
                    leaky_relu(),
                    linear(150, 100),
                    leaky_relu(),
                    linear(100, 52),
                    leaky_relu(),
                    linear(52,1)  ## output size is 1
                ]

net = Model()
net.fit(
    x_train, ## your input features
    y_train, ## your labels
    epochs=60,
    learning_rate=0.005, 
    loss = mse_loss
)

```