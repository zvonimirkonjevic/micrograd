"""A minimal scalar-valued autograd engine.

Defines :class:`Value`, a scalar wrapper that records every operation it takes
part in, building a dynamic computation graph. Calling :meth:`Value.backward`
walks that graph in reverse topological order and applies the chain rule to
populate the ``grad`` field of every node that contributed to the result.
"""

import math
import numpy as np


# ===============================
# Value
# ===============================

class Value:
    """A scalar value and its gradient in a dynamic computation graph.

    Every arithmetic operation on a ``Value`` returns a new ``Value`` that keeps
    references to its operands (``_prev``) and a closure (``_backward``) that
    knows how to push the incoming gradient to those operands. The graph is
    therefore built implicitly during the forward pass.

    Attributes:
        data: The scalar result of the forward pass.
        grad: Derivative of the final output with respect to this value.
            Accumulated during the backward pass, so it must be reset between
            iterations (see ``Module.zero_grad``).
    """


    def __init__(self, data, _children=(), _op=''):
        """Initializes the value and its node in the computation graph.

        Args:
            data: The scalar this node holds.
            _children: Operands this value was computed from. Internal; used to
                reconstruct the graph during backpropagation.
            _op: Symbol of the operation that produced this value. Internal;
                used for debugging and graph visualization.
        """

        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        """Returns a short, human-readable view of the underlying scalar."""

        return f"Value(data={self.data})"

    def __add__(self, other):
        """Adds ``other``, wrapping plain numbers into a ``Value`` first.

        The local derivative of addition is 1 for both operands, so the
        incoming gradient flows through unchanged.

        Args:
            other: A ``Value`` or a plain int/float.

        Returns:
            A new ``Value`` holding the sum.
        """

        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        """Multiplies by ``other``, wrapping plain numbers into a ``Value``.

        The local derivative with respect to one operand is the other operand,
        so each gradient is scaled by its partner's forward value.

        Args:
            other: A ``Value`` or a plain int/float.

        Returns:
            A new ``Value`` holding the product.
        """

        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        """Divides by ``other``, expressed as ``self * other ** -1``.

        Reusing power and multiplication means no extra backward rule is
        needed: the chain rule composes the two existing ones.

        Args:
            other: A ``Value`` or a plain int/float divisor.

        Returns:
            A new ``Value`` holding the quotient.
        """

        return self * other**-1
    
    def __pow__(self, other):
        """Raises this value to a constant power.

        Only numeric exponents are supported, which keeps the backward rule the
        simple power rule ``n * x ** (n - 1)``. A ``Value`` exponent would also
        require a gradient path through the exponent itself.

        Args:
            other: An int or float exponent.

        Returns:
            A new ``Value`` holding ``self.data ** other``.

        Raises:
            AssertionError: If ``other`` is not an int or float.
        """

        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += other * (self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        """Handles ``other * self`` when ``other`` is a plain int or float."""

        return self * other

    def __neg__(self):
        """Negates this value, expressed as multiplication by -1."""

        return self * -1
    
    def __sub__(self, other):
        """Subtracts ``other``, expressed as ``self + (-other)``.

        Args:
            other: A ``Value`` or a plain int/float.

        Returns:
            A new ``Value`` holding the difference.
        """

        return self + (-other)

    def tanh(self):
        """Applies the hyperbolic tangent activation.

        Returns:
            A new ``Value`` in the range (-1, 1). Its backward pass uses the
            identity ``d/dx tanh(x) = 1 - tanh(x) ** 2``, reusing the already
            computed forward output instead of recomputing exponentials.
        """

        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t,(self,), 'tanh')
        def _backward():
            self.grad += (1-t**2) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        """Applies the logistic sigmoid activation.

        Returns:
            A new ``Value`` in the range (0, 1). Its backward pass uses the
            identity ``d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))``.
        """

        x = self.data
        t = 1 / (1 + math.exp(-x))
        out = Value(t, (self,), 'sigmoid')
        def _backward():
            self.grad += (t * (1 - t)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        """Applies the rectified linear unit activation.

        Returns:
            A new ``Value`` equal to ``max(0, self.data)``. The derivative at
            exactly 0 is defined here as 0, the usual convention.
        """

        x = self.data
        t = 0 if x<0 else x
        out = Value(t, (self,), 'relu')
        def _backward():
            self.grad += (0 if x<0 else 1) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        """Applies the natural exponential.

        Returns:
            A new ``Value`` holding ``e ** self.data``. Its backward pass reuses
            the forward output, since ``exp`` is its own derivative.
        """

        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        """Backpropagates gradients from this value through the whole graph.

        Builds a topological ordering of the graph rooted at this node, seeds
        this node's gradient with 1.0 (the derivative of the output with
        respect to itself), then calls each node's local backward rule in
        reverse order. Reverse topological order guarantees a node's gradient
        is fully accumulated before it is propagated to its own operands.

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

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()



# ===============================
# Tensor
# ===============================

class Tensor:
  """An n-dimensional array and its gradient in a dynamic computation graph.

  The array-valued counterpart of :class:`Value`. It follows the same design,
  each operation returns a new ``Tensor`` that remembers its operands
  (``_prev``) and a closure (``_backward``) that pushes the incoming gradient
  back to them, but it batches the arithmetic through NumPy instead of looping
  over scalars.

  This is a naive implementation: the elementwise operations assume both
  operands already share a shape. NumPy will happily broadcast them in the
  forward pass, but the backward pass does not reduce the gradient back to
  each operand's own shape, so a broadcast operation raises during
  ``backward``.

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
      other: A ``Tensor`` or anything ``Tensor`` accepts, of the same shape.

    Returns:
      A new ``Tensor`` holding the elementwise sum.
    """

    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data + other.data, (self, other), "+")

    def _backward():
      self.grad += out.grad * 1.0
      other.grad += out.grad * 1.0
    out._backward = _backward
    return out

  def __mul__(self, other):
    """Multiplies by ``other`` elementwise (Hadamard product, not matmul).

    The local derivative with respect to one operand is the other operand, so
    each gradient is scaled by its partner's forward value.

    Args:
      other: A ``Tensor`` or anything ``Tensor`` accepts, of the same shape.

    Returns:
      A new ``Tensor`` holding the elementwise product.
    """

    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data * other.data, (self, other), "*")

    def _backward():
      self.grad += out.grad * other.data
      other.grad += out.grad * self.data

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
      other: A ``Tensor`` or anything ``Tensor`` accepts, of the same shape.

    Returns:
      A new ``Tensor`` holding the elementwise difference. Its backward pass
      passes the gradient through unchanged to the minuend and negated to the
      subtrahend.
    """

    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data - other.data, (self, other), "-")

    def _backward():
      self.grad += out.grad
      other.grad += -out.grad
    out._backward = _backward
    return out

  def __truediv__(self, other):
    """Divides by ``other`` elementwise.

    Args:
      other: A ``Tensor`` or anything ``Tensor`` accepts, of the same shape.

    Returns:
      A new ``Tensor`` holding the elementwise quotient. Its backward pass
      applies the quotient rule: ``d/da (a / b) = 1 / b`` and
      ``d/db (a / b) = -a / b ** 2``.
    """

    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data / other.data, (self, other), "/")

    def _backward():
      self.grad += out.grad / other.data
      other.grad += -out.grad * self.data / other.data**2
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
