from .nn import Module
from .layers import ValueLayer


class ValueMLP(Module):
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
        self.layers = [ValueLayer(sz[i], sz[i+1]) for i in range(len(layers_sizes))]

    def __call__(self, x):
        """Runs the forward pass through every layer in sequence.

        Args:
            x: Sequence of inputs, of length ``input_size``.

        Returns:
            The final layer's output: a list of ``Value`` objects, or a single
            ``Value`` when the last layer has one neuron.
        """

        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """Returns the parameters of every layer in the network, flattened."""

        return [p for layer in self.layers for p in layer.parameters()]
