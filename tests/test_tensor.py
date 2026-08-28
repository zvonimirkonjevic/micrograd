"""Gradient checks for the array engine against PyTorch.

Mirrors ``test_value.py`` for :class:`~src.engine.Tensor`. Two differences shape
these tests. ``Tensor`` stores ``float32``, so comparisons use a float32
tolerance rather than an exact one, and ``Tensor.backward`` seeds the root with
ones rather than requiring a scalar, which makes a non-scalar root behave as the
sum of its elements. The torch reference therefore calls ``.sum().backward()``.
"""

import numpy as np
import pytest
import torch

from src.engines.tensor import Tensor

RTOL = 1e-6
ATOL = 1e-6

A = [[1.0, 2.0], [3.0, 4.0]]
B = [[0.5, -1.5], [2.0, 0.25]]


def tt(data):
    """Returns a float32 leaf tensor holding ``data`` that requires grad."""

    t = torch.tensor(data, dtype=torch.float32)
    t.requires_grad = True
    return t


def check(out, ref, tensors, refs):
    """Asserts a forward result and every operand gradient match the reference.

    Args:
        out: The ``Tensor`` root of the engine's graph.
        ref: The torch tensor holding the same forward result.
        tensors: The engine leaves whose gradients should be compared.
        refs: The torch leaves, in the same order as ``tensors``.
    """

    out.backward()
    ref.sum().backward()

    np.testing.assert_allclose(out.data, ref.detach().numpy(), rtol=RTOL, atol=ATOL)
    for t, r in zip(tensors, refs):
        np.testing.assert_allclose(t.grad, r.grad.numpy(), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("op", ["add", "mul", "sub", "truediv"])
def test_elementwise_ops_match_torch(op):
    """Checks each elementwise binary operator on two same-shaped operands."""

    import operator

    fn = getattr(operator, op)
    a, b = Tensor(A), Tensor(B)
    at, bt = tt(A), tt(B)

    check(fn(a, b), fn(at, bt), [a, b], [at, bt])


def test_neg_matches_torch():
    """Checks unary negation, whose backward pass flips the gradient's sign."""

    a, at = Tensor(A), tt(A)
    check(-a, -at, [a], [at])


def test_matmul_matches_torch():
    """Checks the matmul rule on a non-square product.

    A ``(2,3) @ (3,4)`` shape is used deliberately: with square operands a
    transposed-in-the-wrong-place gradient still has a valid shape and the test
    would pass anyway.
    """

    x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    y = [[1.0, 0.0, -1.0, 2.0], [0.5, 2.0, 1.0, 0.0], [-1.0, 1.0, 0.5, 3.0]]

    a, b = Tensor(x), Tensor(y)
    at, bt = tt(x), tt(y)

    out = a @ b
    assert out.data.shape == (2, 4)
    check(out, at @ bt, [a, b], [at, bt])
    assert a.grad.shape == (2, 3)
    assert b.grad.shape == (3, 4)


def test_transpose_matches_torch():
    """Checks that transposition sends the gradient back through transposed."""

    x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    a, at = Tensor(x), tt(x)

    out = a.transpose()
    assert out.data.shape == (3, 2)
    check(out, at.T, [a], [at])


def test_chained_graph_matches_torch():
    """Checks a graph mixing matmul with elementwise ops and a reused operand.

    ``a`` feeds both the matmul and the elementwise multiply, so its gradient
    has to accumulate over two paths that arrive with different rules.
    """

    a, b = Tensor(A), Tensor(B)
    at, bt = tt(A), tt(B)

    out = (a @ b) * a - b
    ref = (at @ bt) * at - bt
    check(out, ref, [a, b], [at, bt])


def test_nonscalar_root_behaves_as_sum_of_elements():
    """Pins the documented meaning of calling ``backward`` on a non-scalar root.

    Seeding with ones is equivalent to differentiating ``out.sum()``. This is a
    convention rather than a law, so it is worth asserting directly instead of
    only via the shared ``check`` helper.
    """

    a, b = Tensor(A), Tensor(B)
    out = a * b
    out.backward()

    # d(sum(a * b))/da is just b, elementwise.
    np.testing.assert_allclose(a.grad, np.array(B, dtype=np.float32), rtol=RTOL)
    np.testing.assert_allclose(b.grad, np.array(A, dtype=np.float32), rtol=RTOL)


def test_data_is_cast_to_float32():
    """Checks the dtype contract, including that integer input is not kept."""

    t = Tensor([[1, 2], [3, 4]])
    assert t.data.dtype == np.float32
    assert t.grad.dtype == np.float32
    assert t.grad.shape == t.data.shape


@pytest.mark.parametrize("op", ["add", "mul", "sub", "truediv"])
def test_reflected_operators_match_torch(op):
    """Checks each reflected operator with a plain number on the left.

    ``sub`` and ``truediv`` do not commute, so these also pin the operand
    order: ``2.0 - t`` must not quietly become ``t - 2.0``.
    """

    import operator

    fn = getattr(operator, op)
    a, at = Tensor(A), tt(A)

    check(fn(2.0, a), fn(2.0, at), [a], [at])


def test_ndarray_on_the_left_defers_to_the_reflected_operator():
    """Checks that an ndarray operand does not swallow the ``Tensor``.

    Without ``__array_ufunc__ = None`` NumPy treats a ``Tensor`` as an opaque
    object and broadcasts elementwise against it, producing an object array of
    whole Tensors instead of one ``Tensor`` node. That is silently wrong
    rather than an error, so it is worth asserting the type directly.
    """

    ones = np.ones((2, 2), dtype=np.float32)
    a, at = Tensor(A), tt(A)

    out = ones * a
    assert isinstance(out, Tensor)
    check(out, torch.from_numpy(ones) * at, [a], [at])


@pytest.mark.parametrize("exponent", [2, 3, -1, 0.5])
def test_pow_matches_torch(exponent):
    """Checks the power rule over integer, negative and fractional exponents.

    The exponent is a constant rather than an operand, so the node is unary
    and only ``a`` carries a gradient.
    """

    a, at = Tensor(A), tt(A)

    check(a**exponent, at**exponent, [a], [at])


def test_pow_rejects_a_tensor_exponent():
    """Documents that a ``Tensor`` exponent has no gradient path here."""

    with pytest.raises(AssertionError):
        Tensor(A) ** Tensor(B)


@pytest.mark.parametrize("fn", ["relu", "tanh"])
def test_activations_match_torch(fn):
    """Checks every elementwise nonlinearity on a mix of signs and zero."""

    x = [[-2.0, -0.5], [0.0, 1.5]]
    a, at = Tensor(x), tt(x)

    check(getattr(a, fn)(), getattr(torch, fn)(at), [a], [at])


BROADCAST_SHAPES = [
    ((2, 3), (3,)),    # operand missing a leading axis entirely
    ((2, 3), (1, 3)),  # size-1 row stretched down
    ((2, 1), (2, 3)),  # size-1 column stretched across
    ((1, 1), (2, 3)),  # both axes stretched at once
    ((2, 3), ()),      # scalar operand
]


def arr(shape):
    """Returns a distinct, nonzero float32 array of ``shape``.

    Nonzero everywhere so the division case stays finite, and no two entries
    repeat so a gradient summed over the wrong axis cannot coincidentally match.
    """

    n = int(np.prod(shape))
    return (np.arange(1, n + 1, dtype=np.float32) / 2 + 0.25).reshape(shape)


@pytest.mark.parametrize("op", ["add", "mul", "sub", "truediv"])
@pytest.mark.parametrize("lhs_shape, rhs_shape", BROADCAST_SHAPES)
def test_broadcast_gradients_match_torch(op, lhs_shape, rhs_shape):
    """Checks a broadcast operand's gradient is reduced back to its own shape.

    Broadcasting reuses an operand across every position it was stretched over,
    so the chain rule adds up the gradient arriving at each of those positions.
    Shapes are asserted on top of ``check``: a gradient left at the broadcast
    shape would fail loudly when accumulated into a leaf, but one reduced along
    the wrong axis can still land on a plausible shape.
    """

    import operator

    fn = getattr(operator, op)
    lhs, rhs = arr(lhs_shape), arr(rhs_shape)
    a, b = Tensor(lhs), Tensor(rhs)
    at, bt = tt(lhs), tt(rhs)

    check(fn(a, b), fn(at, bt), [a, b], [at, bt])
    assert a.grad.shape == lhs_shape
    assert b.grad.shape == rhs_shape
