# patrick
> Spongebob's best friend

Tiny neural net library written from scratch with [cupy](https://cupy.dev/) backend (sorry CPU gang). Still under construction. 

<p align="center">
<img src = "images/patrick.png" width = "20%">
</p>


```python
from patrick.nn import NN as nn
from patrick.losses import mse_loss
from patrick.activations import leaky_relu
from patrick.layers  import FCLayer as linear

"""
Load and preprocess your data here 
"""

class Model(nn):
    def __init__(self):
        self.layers =  [
                    linear(64,32),  ## input size is 64
                    leaky_relu(),
                    linear(32, 16),
                    leaky_relu(),
                    linear(16,1)  ## output size is 1
                ]

net = Model()

net.fit(
    x_train,    ## your input features, shape: (num_batches, batch_size, input_size)
    y_train,    ## your labels, shape: (num_batches, batch_size, output_size)
    epochs=60,
    learning_rate=0.005, 
    loss = mse_loss
)
```

"The best way to learn how something works is to make it from scratch" 

                                    -Probably Someone