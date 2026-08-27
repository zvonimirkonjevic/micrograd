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

from src.engine import Tensor

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


def test_reflected_operators_are_not_supported():
    """Documents that ``Tensor`` has no reflected operators, unlike ``Value``.

    A plain number on the left has no ``__mul__`` that understands ``Tensor``,
    and ``Tensor`` defines no ``__rmul__``, so Python gives up with a
    ``TypeError``. Locked down so the gap fails loudly rather than drifting.
    """

    t = Tensor(A)
    assert isinstance(t * 2.0, Tensor)
    with pytest.raises(TypeError):
        2.0 * t


def test_broadcasting_raises_in_backward():
    """Documents the known broadcasting gap as a deliberate limitation.

    The forward pass broadcasts because NumPy does. The backward pass never
    reduces the gradient back to each operand's shape, so accumulating a (2,2)
    gradient into a (2,) leaf fails. This test asserts the current behaviour;
    it is the test to delete when reduction is implemented.
    """

    a = Tensor(A)
    b = Tensor([1.0, 1.0])

    out = a + b
    assert out.data.shape == (2, 2)

    with pytest.raises(ValueError, match="non-broadcastable"):
        out.backward()
