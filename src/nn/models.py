from typing import List

from .nn import Module
from .layers import ValueLayer, TensorLayer


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


class TensorMLP:
  """A multi-layer perceptron on the ``Tensor`` engine.

  The array-valued counterpart of :class:`ValueMLP`, stacking
  :class:`TensorLayer` instead of :class:`ValueLayer`.

  Attributes:
    layers: The stacked layers, in forward order.
  """

  def __init__(self, input_size: int, output_sizes: List[int]):
    """Builds the stack of layers from the requested widths.

    Args:
      input_size: Length of the network's input vector.
      output_sizes: Output width of each layer, in order. Widths are chained
        so that layer ``i`` maps from the previous width to
        ``output_sizes[i]``.
    """

    sz = [input_size] + output_sizes
    self.layers = [TensorLayer(sz[i], sz[i+1]) for i in range(len(output_sizes))]

  def __call__(self, x):
    """Runs the forward pass through every layer in sequence.

    Args:
      x: A 2-D ``Tensor`` of shape ``(batch_size, input_size)``, or a raw
        array-like, which :class:`TensorLayer` wraps and promotes to 2-D. Only
        2-D input is accepted, for the reason given there.

    Returns:
      A ``Tensor`` of shape ``(batch_size, output_sizes[-1])`` holding the
      final layer's output.
    """

    for layer in self.layers:
      x = layer(x)

    return x