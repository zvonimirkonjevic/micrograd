"""Neural network building blocks on top of the scalar autograd engine.

Provides a small module hierarchy - :class:`Neuron`, :class:`Layer` and
:class:`MLP` - where every parameter is a :class:`~src.engine.Value`, so a
forward pass builds the computation graph that ``Value.backward`` later
traverses.
"""

import random
from src.engine import Value

class Module:
    """Base class for anything that owns trainable parameters.

    Subclasses override :meth:`parameters` to expose their own ``Value``
    objects; the gradient bookkeeping is then inherited for free.
    """


    def zero_grad(self):
        """Resets the gradient of every parameter to zero.

        Gradients accumulate across backward passes, so this must be called
        before each optimization step or gradients from previous steps leak
        into the current one.
        """

        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        """Returns the trainable parameters of this module.

        Returns:
            An empty list. Subclasses override this to return their ``Value``
            parameters.
        """

        return []


# 
#   Neurons 
#

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


# 
#   Layers 
#

class Layer(Module):
    """A fully connected layer: a list of independent neurons.

    Attributes:
        neurons: The ``layer_size`` neurons, each seeing the full input vector.
    """

    def __init__(self, input_size, layer_size):
        """Creates ``layer_size`` neurons, each accepting ``input_size`` inputs.

        Args:
            input_size: Length of the input vector fed to every neuron.
            layer_size: Number of neurons, which is the layer's output width.
        """

        self.neurons = [Neuron(input_size) for _ in range(layer_size)]
        
    def __call__(self, x):
        """Runs the forward pass for one input vector.

        Args:
            x: Sequence of inputs, of length ``input_size``.

        Returns:
            A list of output ``Value`` objects, or the single ``Value`` itself
            when the layer has exactly one neuron. Unwrapping the single-output
            case keeps scalar-output networks convenient to use.
        """

        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs
    
    def parameters(self):
        """Returns the parameters of every neuron in the layer, flattened."""

        return [p for neuron in self.neurons for p in neuron.parameters()]


# 
#   Architectures 
#


class MLP(Module):
    """A multi-layer perceptron: layers applied in sequence.

    Attributes:
        layers: The stacked layers, in forward order.
    """

    def __init__(self, input_size, layers_sizes):
        """Builds the stack of layers from the requested widths.

        Args:
            input_size: Length of the network's input vector.
            layers_sizes: Output width of each layer, in order. Widths are
                chained so that layer ``i`` maps from the previous width to
                ``layers_sizes[i]``.
        """

        sz = [input_size] + layers_sizes
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(layers_sizes))]

    def __call__(self, x):
        """Runs the forward pass through every layer in sequence.

        Args:
            x: Sequence of inputs, of length ``input_size``.

        Returns:
            The final layer's output: a list of ``Value`` objects, or a single
            ``Value`` when the last layer has one neuron.
        """

        for layer in self.layers:
            x = Layer(x)
        return x

    def parameters(self):
        """Returns the parameters of every layer in the network, flattened."""

        return [p for layer in self.layers for p in layer.parameters()]
