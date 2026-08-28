from .neurons import Neuron
from .nn import Module


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