import random
import numpy as np

from .nn import Module
from src.engines.value import Value
from src.engines.tensor import Tensor


class ValueNeuron(Module):
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


class TensorNeuron:
  """A single neuron computing ``tanh(w . x + b)`` on the ``Tensor`` engine.

  The array-valued counterpart of :class:`ValueNeuron`. It holds the same
  parameters, but as two ``Tensor`` objects instead of a list of scalar
  ``Value`` objects, so the dot product runs as one NumPy multiply followed by
  a :meth:`Tensor.sum` rather than a Python loop.

  Weights and bias are initialized uniformly in [-1, 1].

  Attributes:
    w: A 1-D ``Tensor`` of ``input_size`` weights.
    b: A 0-d ``Tensor`` holding the bias.
  """

  def __init__(self, input_size: int):
    """Initializes weights and bias with random values in [-1, 1].

    Args:
      input_size: Number of inputs this neuron accepts, which is also the
        number of weights created.
    """

    self.w = Tensor(np.random.uniform(low=-1, high=1, size=(input_size, 1)).flatten())
    self.b = Tensor(np.random.uniform(low=-1, high=1))

  def __call__(self, x):
    """Runs the forward pass for one input vector.

    Args:
      x: A ``Tensor`` or anything ``Tensor`` accepts, of length
        ``input_size``. Raw array-likes are wrapped so callers can pass plain
        lists.

    Returns:
      A 0-d ``Tensor`` holding the activated output. :meth:`Tensor.sum`
      collapses the dot product to 0-d and the bias is 0-d as well, so the
      output stays 0-d. Reshaping it afterwards would leave ``data`` and the
      already allocated ``grad`` with different shapes, which breaks the
      backward pass.
    """

    if not isinstance(x, Tensor):
      x = Tensor(x)
    act = (x * self.w).sum() + self.b
    out = act.tanh()
    return out