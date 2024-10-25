import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


# 
#   Neurons 
#


class Neuron(Module):
    def __init__(self, input_size):
        self.w = [Value(random.uniform(-1,1)) for _ in range(input_size)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]
    

class RNNNeuron(Module):
    def __init__(self, input_size, non_lin=True):
        self.w_xh = [Value(random.unifrom(-0.1, 0.1)) for _ in range(input_size)]
        self.w_hh = [Value(random.unifrom(-0.1, 0.1)) for _ in range(input_size)]
        self.b = Value(0)
        self.h = Value(0)
        self.non_lin = non_lin
    
    def __call__(self, x, h_prev):
        h = sum(wi * xi for wi, xi in zip(self.w_xh, x)) + sum(wj * hj for wj, hj in zip(self.w_hh, h_prev)) + self.b
        h = h.tanh() if self.non_lin else h
        self.h = h 
        return h
    
    def parameters(self):
        return self.w_xh + self.w_hh + [self.b]


# 
#   Layers 
#


class Layer(Module):
    def __init__(self, input_size, layer_size):
        self.neurons = [Neuron(input_size) for _ in range(layer_size)]
        
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    

class RNNLayer(Module):
    def __init__(self, input_size, layer_size, **kwargs):
        self.neuron = RNNNeuron(input_size, **kwargs)
        self.outs = layer_size

    def __call__(self, x, h_prev):
        h_new = []
        for i in range(len(x)):
            h = self.neuron(x[i], h_prev)
            h_new.append(h)
        return h_new

    def parameters(self):
        return self.neuron.parameters()


# 
#   Architectures 
#


class MLP(Module):
    def __init__(self, input_size, layers_sizes):
        sz = [input_size] + layers_sizes
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(layers_sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = Layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    

class RNN(Module):
    def __init__(self, input_size, layers_size):
        sz = [input_size] = layers_size
        self.layers = [RNNLayer(sz[i], sz[i+1], non_lin=i != len(layers_size)-1) for i in range(len(layers_size))]

    def __call__(self, x, h_prevs):
        h_new_all = []
        for layer_idx, layer in enumerate(self.layers):
            h_prev = h_prevs[layer_idx]
            h_new = layer(x, h_prev)
            h_new_all.append(h_new)
            x = h_new
        
        return x, h_new_all

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]