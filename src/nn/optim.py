import numpy as np


class SGD:
    """Stochastic gradient descent over a fixed list of parameters.

    The optimizer holds no state of its own beyond the learning rate: each
    step moves every parameter downhill along its own gradient, which is what
    makes plain SGD the cheapest optimizer to implement.

    Attributes:
        parameters: The ``Tensor`` parameters this optimizer updates.
        lr: Learning rate, the step size taken along the negative gradient.
    """

    def __init__(self, parameters, lr=0.01):
        """Binds the optimizer to the parameters it will update.

        Args:
            parameters: Iterable of ``Tensor`` parameters, typically the result
                of a module's ``parameters()`` call.
            lr: Learning rate. Too large a value overshoots the minimum and the
                loss diverges, too small a value trains slowly.
        """

        self.parameters = parameters
        self.lr = lr

    def step(self):
        """Moves every parameter one learning-rate step down its gradient.

        Assumes ``backward()`` has already populated ``p.grad`` for this batch;
        calling it twice without a fresh backward pass applies the same update
        again.
        """

        for p in self.parameters:
            p.data += -self.lr * p.grad

    def zero_grad(self):
        """Resets the gradient of every parameter to zero.

        Gradients accumulate across backward passes, so this must be called
        before each backward pass or gradients from previous steps leak into
        the current one.
        """

        for p in self.parameters:
            p.grad = np.zeros_like(p.data)
