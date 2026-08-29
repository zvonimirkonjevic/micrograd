import numpy as np

from .neurons import ValueNeuron
from .nn import Module
from src.engines.tensor import Tensor


class ValueLayer(Module):
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

        self.neurons = [ValueNeuron(input_size) for _ in range(layer_size)]
        
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


class TensorLayer:
  """A fully connected layer on the ``Tensor`` engine.

  The array-valued counterpart of :class:`ValueLayer`. Where ``ValueLayer``
  keeps a list of independent neurons, this packs the whole layer into one
  weight matrix and one bias vector, so the forward pass is a single matrix
  multiply followed by one activation, instead of one dot product and one
  ``tanh`` per neuron.

  Attributes:
    w: The ``(input_size, output_size)`` weight matrix.
    b: The ``(output_size,)`` bias vector, broadcast across the batch.
    activation: The nonlinearity applied to the layer's outputs, or ``None``
      for a purely affine layer.
  """

  def __init__(self, input_size: int, output_size: int, activation=Tensor.tanh):
    """Initializes the weight matrix and bias with random values in [-1, 1].

    Args:
      input_size: Length of the input vector fed to the layer.
      output_size: The layer's output width.
      activation: A ``Tensor`` method applied to the layer's outputs, such as
        ``Tensor.tanh`` or ``Tensor.relu``. Pass ``None`` for an affine layer,
        which is what an output layer needs when the targets fall outside the
        activation's range.
    """

    self.w = Tensor(np.random.uniform(-1, 1, size=(input_size, output_size)))
    self.b = Tensor(np.random.uniform(-1, 1, size=output_size))
    self.activation = activation

  def __call__(self, x):
    """Runs the forward pass for a batch of input vectors.

    Args:
      x: A 2-D ``Tensor`` of shape ``(batch_size, input_size)``. Only 2-D
        input is accepted: :meth:`Tensor.__matmul__` requires two 2-D
        operands, and a 1-D ``Tensor`` is passed through as given, which fails
        in the backward pass. Raw array-likes are wrapped so callers can pass
        plain lists, and a 1-D one is promoted to a single-row batch of shape
        ``(1, input_size)``.

    Returns:
      A ``Tensor`` of shape ``(batch_size, output_size)`` holding the layer's
      outputs, activated unless ``activation`` is ``None``. The activation
      runs on the whole matrix at once, which is where :class:`ValueLayer`
      applies ``tanh`` inside every neuron.
    """

    if not isinstance(x, Tensor):
      x = Tensor(np.atleast_2d(x))
    acts = x @ self.w + self.b
    if self.activation is None:
      return acts
    return self.activation(acts)

  def parameters(self):
    """Returns the weight matrix followed by the bias vector."""

    return [self.w, self.b]