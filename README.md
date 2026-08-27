# micrograd

Two automatic differentiation engines that share one design: a scalar engine
where every number is a graph node, and an array engine where every tensor is.
Written to find out precisely where the scalar formulation stops scaling and
what has to change when it does.

Reverse-mode autodiff is usually met as a black box behind `loss.backward()`.
The mechanism underneath is small enough to hold in your head: build a graph
during the forward pass, sort it topologically, walk it backwards applying the
chain rule. What is genuinely hard is not the algorithm but the bookkeeping,
and the bookkeeping only becomes visible once values stop being scalars. That
is the transition this repo is built around.

## The shared design

Both engines use the same three-part contract, and nothing else:

1. **An operation returns a new node**, never mutates its inputs. That node
   holds the forward result plus references to the operands it came from
   (`_prev`), so the graph is a byproduct of computing, not a separate
   structure anyone has to declare.
2. **Each node owns a closure** (`_backward`) that knows only its own local
   derivative and how to push an incoming gradient to its immediate operands.
   No node knows anything about the graph it sits in.
3. **`backward()` supplies the ordering.** It topologically sorts from the
   root, seeds the root gradient, then walks the sort in reverse. Reverse
   topological order is the whole trick: it guarantees a node's gradient is
   fully accumulated from every downstream path before that node propagates
   anything to its own operands.

Because the local rules know nothing about ordering and the ordering knows
nothing about the rules, adding an operation means writing one closure. That
property is what makes the jump from scalars to arrays a matter of changing
what a node holds, not how anything traverses.

## What each engine supports

| | `Value` (scalar) | `Tensor` (array) |
| --- | --- | --- |
| Backing | Python float | `np.ndarray`, float32 |
| Arithmetic | `+ - * / **` , unary `-` | `+ - * /` , unary `-` |
| Linear algebra | not applicable | `@` matmul, `transpose` |
| Activations | `tanh`, `sigmoid`, `relu`, `exp` | none yet |
| Gradient | float, seeded `1.0` | array shaped like `data`, seeded with ones |
| Built on it | `Neuron`, `Layer`, `MLP` | nothing yet |

## The scalar engine

```python
from src.engine import Value

x = Value(2.0)
y = Value(-3.0)
z = (x * y + x.exp()).tanh()

z.backward()
print(z.data, x.grad, y.grad)  # 0.883 0.967 0.441
```

Two rules are written in terms of others rather than getting their own
closure: division is `self * other ** -1` and subtraction is `self + (-other)`.
The chain rule composes the existing power and multiplication rules for free,
so there is no second derivation to get wrong.

`__pow__` deliberately accepts only numeric exponents and asserts on anything
else. A `Value` exponent would need a gradient path through the exponent
itself, `d/dn (x ** n) = x ** n * ln(x)`, which is a different rule wearing the
same syntax.

## The array engine

`Tensor` keeps the contract above and swaps floats for `np.ndarray`. Most
operations survive the move unchanged, since elementwise arithmetic is just the
scalar rule applied in parallel. Matmul is where it gets interesting, because it
is the first operation whose backward pass is not the forward pass with
different numbers.

For `C = A @ B`, the gradients have to come back out shaped like `A` and `B`,
which forces the transposes:

```
dA = dC @ B.T        (m, k) = (m, n) @ (n, k)
dB = A.T @ dC        (k, n) = (k, m) @ (m, n)
```

There is exactly one way to arrange each product that produces the right shape,
which is a useful property: shape agreement is a strong check on whether the
rule is correct.

```python
from src.engine import Tensor

a = Tensor([[1.0, 2.0], [3.0, 4.0]])
w = Tensor([[0.5], [-0.5]])

out = a @ w
out.backward()
print(a.grad)  # shaped like a, not a scalar
```

`backward()` seeds the root with ones, which implicitly treats a non-scalar
root as the sum of its elements. Call it on a scalar loss to get the gradients
you actually want.

## Neural nets on the scalar engine

`src/nn.py` stacks `Neuron` into `Layer` into `MLP`, all sharing a `Module`
base that provides `parameters()` and `zero_grad()`. Nothing in these classes
touches gradients directly; they only build expressions, and the engine handles
the rest.

```python
from src.nn import MLP

model = MLP(3, [4, 4, 1])          # 3 inputs, two hidden layers, scalar output

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

for _ in range(50):
    preds = [model(x) for x in xs]
    loss = sum((p - t)**2 for t, p in zip(ys, preds))

    model.zero_grad()
    loss.backward()

    for p in model.parameters():
        p.data -= 0.05 * p.grad

print(loss.data)  # ~0.02
```

`zero_grad()` is not optional. Gradients accumulate rather than overwrite,
which is what makes a node reachable by several paths sum its contributions
correctly, and it is also what makes last step's gradients leak into this one
if nobody clears them.

`Neuron.__call__` seeds its dot product with the bias, `sum(w*x for ..., self.b)`,
which folds the bias in and keeps the accumulator a `Value` from the first
addition on.

## Derivations

The notebooks work each engine out by hand before any of it becomes library
code, with graphviz renderings of the computation graphs at each stage.

- `notebooks/micrograd_from_scratch.ipynb`: finite-difference intuition,
  manually assigned gradients on a four-node graph, then the generalization to
  automatic traversal.
- `notebooks/micrograd_tensor_addition.ipynb`: the same progression for arrays,
  including working the matmul gradient shapes out by hand on a `(2,1) @ (1,3)`
  graph before automating them.

## Install

```bash
uv sync
```

Requires Python 3.14. NumPy is the only runtime dependency, used solely as an
array container and BLAS backend; every derivative here is hand-written.
