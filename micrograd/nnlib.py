import random
from micrograd.engine import Value

class Neuron:
    def __init__(self, input_size):
        self.w = [Value(random.uniform(-1,1)) for _ in range(input_size)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

class Layer:
    def __init__(self, input_size, layer_size):
        self.neurons = [Neuron(input_size) for _ in range(layer_size)]
        
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs


