# micrograd

Two reverse-mode autodiff engines and a small neural net library on top of them.
`Value` is scalar-valued: every number is a node in a dynamically constructed
DAG, so a single neuron gets chopped into its individual adds and multiplies.
`Tensor` is the array-valued counterpart: same design, but each node holds an
`np.ndarray` and the arithmetic is batched through NumPy. `src/nn/` stacks
`Neuron` into `Layer` into `MLP` with a PyTorch-like API, once per engine, plus
an `SGD` optimizer.

Ignoring docstrings, `Value` is about 91 lines, `Tensor` about 123, and the nn
package about 94. NumPy is the only library the engines and the nn package
import, and it is used purely as an array container and BLAS backend: every
derivative here is hand-written. graphviz, matplotlib and scikit-learn are
dependencies of the notebooks, not of the library. PyTorch is pulled in only by
the test group, as a reference to check gradients against.

A reimplementation of [karpathy/micrograd](https://github.com/karpathy/micrograd),
extended with the array engine. Educational, not fast.

### Install

```bash
uv sync
```

Requires Python 3.14. The notebooks additionally want a system `graphviz`
binary for the graph renderings.

### Layout

```
src/engines/value.py   the scalar engine
src/engines/tensor.py  the array engine
src/nn/nn.py           Module, the parameters()/zero_grad() base class
src/nn/neurons.py      ValueNeuron, TensorNeuron
src/nn/layers.py       ValueLayer, TensorLayer
src/nn/models.py       ValueMLP, TensorMLP
src/nn/optim.py        SGD
```

Every nn class comes in a `Value` and a `Tensor` flavour, and the two are not
interchangeable: a `ValueMLP` consumes and returns `Value` objects, a
`TensorMLP` consumes and returns `Tensor` objects.

### Example usage: the scalar engine

```python
from src.engines.value import Value

x = Value(2.0)
y = Value(-3.0)
z = (x * y + x.exp()).tanh()

z.backward()
print(z.data)   # 0.8830, the forward pass
print(x.grad)   # 0.9672, i.e. dz/dx
print(y.grad)   # 0.4408, i.e. dz/dy
```

Supported: `+ - * / **`, unary `-`, the reflected forms `__radd__`, `__rsub__`,
`__rmul__`, `__rtruediv__` (so `2 - x` and `sum(...)` over `Value`s work), and
`tanh`, `sigmoid`, `relu`, `exp`.

Division and subtraction get no backward rule of their own. They are written as
`self * other ** -1` and `self + (-other)`, and the chain rule composes the
existing power and multiplication rules for free. `__pow__` asserts on a
non-numeric exponent: a `Value` exponent needs a second gradient path,
`d/dn (x ** n) = x ** n * ln(x)`, which is a different rule wearing the same
syntax.

### Example usage: the array engine

```python
from src.engines.tensor import Tensor

a = Tensor([[1.0, 2.0], [3.0, 4.0]])
w = Tensor([[0.5], [-0.5]])

out = a @ w
out.backward()
print(a.grad)   # [[ 0.5 -0.5], [ 0.5 -0.5]], shaped like a
print(w.grad)   # [[4.], [6.]],               shaped like w
```

Supported: `+ - * / **` elementwise, `@` matmul, unary `-`, the reflected forms
`__radd__`, `__rsub__`, `__rmul__`, `__rtruediv__`, `transpose`, `sum`, and the
activations `relu` and `tanh`. Data is cast to `float32`.

The elementwise ops broadcast the way NumPy does, which the backward pass has
to undo. Broadcasting reuses an operand across the positions it was stretched
over, so the chain rule sums the incoming gradient over those positions: every
elementwise backward routes its gradient through `unbroadcast`, which sums away
the axes NumPy prepended and then sums, with `keepdims`, the axes the operand
had as length 1. Without it a gradient comes back shaped like the output rather
than like the operand, and `grad += ...` broadcasts it into the wrong shape
instead of failing.

The reflected forms come with `__array_ufunc__ = None`. Without it a NumPy
array on the left wins the dispatch, treats the `Tensor` as an opaque object
and broadcasts elementwise against it, so `np.ones((2, 2)) * t` returns an
object array holding four copies of the whole tensor rather than one node.
Opting out of the ufunc protocol makes NumPy return `NotImplemented`, and
Python falls back to `__rmul__`.

Matmul is the first operation whose backward pass is not the forward pass with
different numbers. For `C = A @ B`:

```
dA = dC @ B.T        (m, k) = (m, n) @ (n, k)
dB = A.T @ dC        (k, n) = (k, m) @ (m, n)
```

Only one arrangement of each product yields the right shape, which makes shape
agreement a decent check on whether the rule is correct. Both operands must be
2-D; NumPy's 1-D matmul rules would silently produce mis-shaped gradients.

`Tensor.backward()` seeds the root with `np.ones_like(data)`, which implicitly
treats a non-scalar root as the sum of its elements. Call it on a scalar loss to
get the gradients you actually want.

### Training a neural net: the scalar engine

`ValueMLP(3, [4, 4, 1])` is 3 inputs, two hidden layers of 4, and a scalar
output: 41 parameters, each an individual `Value`.

```python
from src.nn.models import ValueMLP

model = ValueMLP(3, [4, 4, 1])

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

for _ in range(50):
    preds = [model(x) for x in xs]
    loss = sum((p - t)**2 for t, p in zip(ys, preds))

    model.zero_grad()
    loss.backward()

    for p in model.parameters():
        p.data -= 0.05 * p.grad

print(loss.data)  # ~0.01 after 50 steps of SGD
```

`zero_grad()` is not optional. Gradients accumulate rather than overwrite, which
is what lets a node reachable by several paths sum its contributions correctly,
and it is also what makes the last step's gradients leak into this one if nobody
clears them.

### Training a neural net: the array engine

The same network on the `Tensor` engine is 6 parameter tensors rather than 41
scalars: one weight matrix and one bias vector per layer. The whole batch goes
through in one forward pass instead of a Python loop over the rows, and `SGD`
takes over the parameter update.

```python
from src.engines.tensor import Tensor
from src.nn.models import TensorMLP
from src.nn.optim import SGD

model = TensorMLP(3, [4, 4, 1])
optimizer = SGD(model.parameters(), lr=0.05)

xs = Tensor([[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]])
ys = Tensor([[1.0], [-1.0], [-1.0], [1.0]])

for _ in range(50):
    preds = model(xs)
    loss = ((preds - ys) ** 2).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(loss.data)  # a few hundredths after 50 steps of SGD
```

`.sum()` is doing real work here. `backward()` seeds the root with ones, so
calling it on the `(4, 1)` matrix of per-sample errors would differentiate the
sum of those errors implicitly, with nothing naming the loss. Summing first
makes the scalar explicit.

`SGD` is deliberately thin: it holds the parameter list and a learning rate, and
`step()` is one `p.data += -lr * p.grad` per parameter. It exists to move the
update rule out of the training loop, not because plain SGD needs any state. Its
`zero_grad()` resets to `np.zeros_like(p.data)`, which is why it belongs to the
`Tensor` engine, where `Module.zero_grad()` resets to the scalar `0.0` for
`Value` parameters.

### How it works

Both engines use the same three-part contract:

1. **An operation returns a new node** and never mutates its inputs. That node
   holds the forward result plus references to its operands (`_prev`), so the
   graph is a byproduct of computing rather than a structure anyone declares.
2. **Each node owns a closure** (`_backward`) that knows only its own local
   derivative and how to push an incoming gradient to its immediate operands.
   No node knows anything about the graph it sits in.
3. **`backward()` supplies the ordering.** It topologically sorts from the root,
   seeds the root gradient, then walks the sort in reverse. Reverse topological
   order guarantees a node's gradient is fully accumulated from every downstream
   path before that node propagates anything to its own operands.

Because the local rules know nothing about ordering and the ordering knows
nothing about the rules, adding an operation means writing one closure, and
going from scalars to arrays changes what a node holds rather than how anything
is traversed.

### Limitations

- `Tensor.__matmul__` requires two 2-D operands. It does not broadcast over a
  batch dimension, and NumPy's 1-D matmul rules would silently produce
  mis-shaped gradients, so the elementwise ops broadcast but this one does not.
- `TensorLayer` takes its activation as a constructor argument, defaulting to
  `Tensor.tanh`, and applies it to the whole output matrix at once. Where
  `ValueLayer` hardcodes `tanh` inside each neuron, `TensorMLP` gives every
  layer the same activation, including the last one, so a regression target
  outside `[-1, 1]` needs `activation=None` on an output layer built directly
  as a `TensorLayer`.
- `Tensor.sum()` collapses the entire array to 0-d. There is no `axis`
  argument, so a per-sample loss cannot be reduced along one dimension only.
- The scalar engine allocates a node per arithmetic operation. It is meant to be
  read, not to train anything of size.

### Notebooks

Each engine is worked out by hand before it becomes library code, with graphviz
renderings of the computation graph at each stage.

- `notebooks/micrograd_from_scratch.ipynb`: finite-difference intuition,
  manually assigned gradients on a four-node graph, then the generalization to
  automatic traversal.
- `notebooks/micrograd_tensor_addition.ipynb`: the same progression for arrays,
  including deriving the matmul gradient shapes by hand on a `(2,1) @ (1,3)`
  graph before automating them, then an MLP and a gradient descent loop.
- `notebooks/micrograd_engine_benchmark.ipynb`: the two engines trained on the
  same scikit-learn datasets, matched parameter for parameter, timed against
  each other, and finished with loss curves and a decision boundary on
  `make_circles`.

### Running tests

The tests use [PyTorch](https://pytorch.org/) as a reference: every test builds
the same expression twice, once with this engine and once with `torch`, then
asserts the forward values and the gradients agree. Torch is an opt-in
dependency group, so a plain `uv sync` does not pull it in.

```bash
uv run --group test pytest
```

`tests/test_value.py` covers the arithmetic operators, the reflected forms, each
activation, the power rule's rejection of a `Value` exponent, gradient
accumulation through a diamond graph, the topological sort's deduplication of a
shared subexpression, and a hand-built neuron. `tests/test_tensor.py`
covers the elementwise ops, matmul on a deliberately non-square `(2,3) @ (3,4)`
product where a misplaced transpose cannot accidentally still typecheck,
transpose, the float32 cast, the sum-of-elements meaning of a non-scalar root,
each elementwise op against every shape broadcasting can stretch, the power
rule over integer, negative and fractional exponents, both activations, the
reflected forms with a number on the left, and the ndarray-on-the-left dispatch
that `__array_ufunc__` governs. One test pins a known divergence rather than
correct behaviour: `relu`'s subgradient at exactly 0, which is 1 here and 0 in
PyTorch.

### License

MIT
