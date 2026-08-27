# micrograd

Two reverse-mode autodiff engines and a small neural net library on top of them.
`Value` is scalar-valued: every number is a node in a dynamically constructed
DAG, so a single neuron gets chopped into its individual adds and multiplies.
`Tensor` is the array-valued counterpart: same design, but each node holds an
`np.ndarray` and the arithmetic is batched through NumPy. `src/nn.py` stacks
`Neuron` into `Layer` into `MLP` with a PyTorch-like API.

Ignoring docstrings, `Value` is about 90 lines, `Tensor` about 74, and the nn
library about 36. NumPy is the only dependency and is used purely as an array
container and BLAS backend: every derivative here is hand-written.

A reimplementation of [karpathy/micrograd](https://github.com/karpathy/micrograd),
extended with the array engine. Educational, not fast.

### Install

```bash
uv sync
```

Requires Python 3.14.

### Example usage: the scalar engine

```python
from src.engine import Value

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
from src.engine import Tensor

a = Tensor([[1.0, 2.0], [3.0, 4.0]])
w = Tensor([[0.5], [-0.5]])

out = a @ w
out.backward()
print(a.grad)   # [[ 0.5 -0.5], [ 0.5 -0.5]], shaped like a
print(w.grad)   # [[4.], [6.]],               shaped like w
```

Supported: `+ - * /` elementwise, `@` matmul, unary `-`, and `transpose`. Data
is cast to `float32`. There are no reflected operators and no activations yet,
so `t * 2.0` works but `2.0 * t` raises `TypeError`.

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

### Training a neural net

`MLP(3, [4, 4, 1])` is 3 inputs, two hidden layers of 4, and a scalar output:
41 parameters, each an individual `Value`.

```python
from src.nn import MLP

model = MLP(3, [4, 4, 1])

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

- **`Tensor` does not handle broadcasting.** The elementwise ops assume both
  operands already share a shape. NumPy broadcasts happily in the forward pass,
  but the backward pass never reduces the gradient back to each operand's own
  shape, so `Tensor([[1.,2.],[3.,4.]]) + Tensor([1.,1.])` succeeds forward and
  then raises `ValueError: non-broadcastable output operand` in `backward()`.
  This is deliberate: broadcasting gradients is a separate idea from the chain
  rule, and folding it in would obscure the part this repo is about.
- `Tensor` has no activations, so no neural nets are built on it yet. `src/nn.py`
  runs entirely on `Value`.
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
  graph before automating them.

### License

MIT
