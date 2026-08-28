import numpy as np


class Tensor:
  """An n-dimensional array and its gradient in a dynamic computation graph.

  The array-valued counterpart of :class:`Value`. It follows the same design,
  each operation returns a new ``Tensor`` that remembers its operands
  (``_prev``) and a closure (``_backward``) that pushes the incoming gradient
  back to them, but it batches the arithmetic through NumPy instead of looping
  over scalars.

  The elementwise operations broadcast the way NumPy does. Broadcasting
  reuses an operand across the positions it was stretched over, so the chain
  rule sums the incoming gradient over those positions: each backward pass
  routes its gradient through :func:`unbroadcast` to reduce it back to the
  operand's own shape. ``__matmul__`` is the exception, it requires two 2-D
  operands and broadcasts nothing.

  Attributes:
    data: The forward-pass array, always ``float32``.
    grad: Derivative of the final output with respect to this tensor, shaped
      like ``data``. Accumulated during the backward pass, so it must be reset
      between iterations.
  """

  def __init__(self, data, _children=(), _op=''):
    """Initializes the tensor and its node in the computation graph.

    Args:
      data: Anything ``np.array`` accepts: a scalar, a nested sequence or an
        existing array. It is copied and cast to ``float32``.
      _children: Operands this tensor was computed from. Internal; used to
        reconstruct the graph during backpropagation.
      _op: Symbol of the operation that produced this tensor. Internal; used
        for debugging and graph visualization.
    """

    self.data = np.array(data, dtype=np.float32)
    self.grad = np.zeros_like(self.data)
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op

  def __repr__(self):
    """Returns a short, human-readable view of the underlying array."""

    return f"Tensor({self.data})"

  def __add__(self, other):
    """Adds ``other`` elementwise, wrapping raw array-likes into a ``Tensor``.

    The local derivative of addition is 1 for both operands, so the incoming
    gradient flows through unchanged.

    Args:
      other: A ``Tensor`` or anything ``Tensor`` accepts, broadcastable
        against this tensor's shape.

    Returns:
      A new ``Tensor`` holding the elementwise sum.
    """

    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data + other.data, (self, other), "+")

    def _backward():
      self.grad += unbroadcast(out.grad, self.data.shape)
      other.grad += unbroadcast(out.grad, other.data.shape)
    out._backward = _backward
    return out

  def __mul__(self, other):
    """Multiplies by ``other`` elementwise (Hadamard product, not matmul).

    The local derivative with respect to one operand is the other operand, so
    each gradient is scaled by its partner's forward value.

    Args:
      other: A ``Tensor`` or anything ``Tensor`` accepts, broadcastable
        against this tensor's shape.

    Returns:
      A new ``Tensor`` holding the elementwise product.
    """

    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data * other.data, (self, other), "*")

    def _backward():
      self.grad += unbroadcast(out.grad * other.data, self.data.shape)
      other.grad += unbroadcast(out.grad * self.data, other.data.shape)

    out._backward = _backward
    return out

  def __matmul__(self, other):
    """Matrix-multiplies this tensor by ``other``.

    For ``C = A @ B`` the chain rule gives ``dA = dC @ B.T`` and
    ``dB = A.T @ dC``. Both operands must be 2-D: the transposes above assume
    it, and NumPy's 1-D matmul rules would silently produce mis-shaped
    gradients.

    Args:
      other: A ``Tensor`` or anything ``Tensor`` accepts, with a leading
        dimension matching this tensor's trailing one.

    Returns:
      A new ``Tensor`` holding the matrix product.
    """

    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data @ other.data, (self, other), "@")

    def _backward():
      self.grad += out.grad @ other.data.T
      other.grad += self.data.T @ out.grad
    out._backward = _backward
    return out

  def __neg__(self):
    """Negates this tensor elementwise.

    Returns:
      A new ``Tensor`` holding ``-self.data``. Its backward pass flips the
      sign of the incoming gradient.
    """

    out = Tensor(self.data * -1, (self,), "neg")

    def _backward():
      self.grad += -out.grad
    out._backward = _backward
    return out

  def __sub__(self, other):
    """Subtracts ``other`` elementwise.

    Args:
      other: A ``Tensor`` or anything ``Tensor`` accepts, broadcastable
        against this tensor's shape.

    Returns:
      A new ``Tensor`` holding the elementwise difference. Its backward pass
      passes the gradient through unchanged to the minuend and negated to the
      subtrahend.
    """

    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data - other.data, (self, other), "-")

    def _backward():
      self.grad += unbroadcast(out.grad, self.data.shape)
      other.grad += unbroadcast(-out.grad, other.data.shape)
    out._backward = _backward
    return out

  def __truediv__(self, other):
    """Divides by ``other`` elementwise.

    Args:
      other: A ``Tensor`` or anything ``Tensor`` accepts, broadcastable
        against this tensor's shape.

    Returns:
      A new ``Tensor`` holding the elementwise quotient. Its backward pass
      applies the quotient rule: ``d/da (a / b) = 1 / b`` and
      ``d/db (a / b) = -a / b ** 2``.
    """

    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data / other.data, (self, other), "/")

    def _backward():
      self.grad += unbroadcast(out.grad / other.data, self.data.shape)
      other.grad += unbroadcast(-out.grad * self.data / other.data**2, other.data.shape)
    out._backward = _backward
    return out

  def transpose(self):
    """Reverses the tensor's axes.

    Returns:
      A new ``Tensor`` holding the transposed array. Transposition only
      relabels positions, so its backward pass transposes the incoming
      gradient back.
    """

    out = Tensor(self.data.T, (self,), "T")

    def _backward():
      self.grad += out.grad.T
    out._backward = _backward
    return out

  def backward(self):
    """Backpropagates gradients from this tensor through the whole graph.

    Builds a topological ordering of the graph rooted at this node, seeds this
    node's gradient with ones (the derivative of each output element with
    respect to itself), then calls each node's local backward rule in reverse
    order. Reverse topological order guarantees a node's gradient is fully
    accumulated before it is propagated to its own operands.

    Seeding with ones means a non-scalar root is implicitly treated as the sum
    of its elements, so call this on a scalar loss to get the usual gradients.
    Gradients accumulate rather than overwrite, so callers must zero them
    between optimization steps.
    """

    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)

    build_topo(self)

    self.grad = np.ones_like(self.data)
    for node in reversed(topo):
      node._backward()

  def relu(self):
    out = Tensor(np.maximum(self.data, 0), (self,), "ReLU")
    def _backward():
      self.grad += out.grad * (self.data > 0)
    out._backward = _backward
    return out

  def tanh(self):
    out = Tensor(np.tanh(self.data), (self,), "tanh")
    def _backward():
      self.grad += out.grad * (1 - out.data ** 2)
    out._backward = _backward
    return out

  def sum(self):
    """Sums every element into a scalar tensor.

    This is the usual way to collapse a network's outputs into the single
    scalar that :meth:`backward` expects.

    Returns:
      A new 0-d ``Tensor`` holding the total. Each input element contributes
      to the sum exactly once, so the local derivative is 1 everywhere and the
      backward pass copies the scalar incoming gradient into every position.
      NumPy broadcasts the 0-d ``out.grad`` across ``self.grad`` to do so.
    """

    out = Tensor(self.data.sum(), (self,), "sum")

    def _backward():
      self.grad += out.grad
    out._backward = _backward
    return out


# ================================
# Helpers
# ================================

def unbroadcast(grad, shape):
  """Reduces a broadcasted gradient back to the original shape.

  Args:
    grad: The broadcasted gradient, shaped like the output of the forward
      operation.
    shape: The original shape of the operand that was broadcasted.

  Returns:
    The reduced gradient, shaped like ``shape``.
  """

  while len(grad.shape) > len(shape):
    grad = grad.sum(axis=0)
  for axis, size in enumerate(shape):
    if size == 1:
      grad = grad.sum(axis=axis, keepdims=True)
  return grad
