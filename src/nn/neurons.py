import random

from .nn import Module
from src.engines.value import Value


class Neuron(Module):
    """A single neuron computing ``tanh(w . x + b)``.

    Weights and bias are initialized uniformly in [-1, 1].

    Attributes:
        w: One weight ``Value`` per input.
        b: The bias ``Value``.
    """

    def __init__(self, input_size):
        """Initializes weights and bias with random values in [-1, 1].

        Args:
            input_size: Number of inputs this neuron accepts, which is also the
                number of weights created.
        """

        self.w = [Value(random.uniform(-1,1)) for _ in range(input_size)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        """Runs the forward pass for one input vector.

        The bias is used as the starting value of the sum, which both folds it
        into the dot product and keeps the accumulator a ``Value`` from the
        first addition onwards.

        Args:
            x: Sequence of inputs, of length ``input_size``.

        Returns:
            A ``Value`` holding the activated output.
        """

        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        """Returns the weights followed by the bias."""

        return self.w + [self.b]