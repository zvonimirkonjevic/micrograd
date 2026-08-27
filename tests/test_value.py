"""Gradient checks for the scalar engine against PyTorch.

Every test builds the same expression twice, once with :class:`~src.engine.Value`
and once with a ``torch`` tensor, then asserts the forward results and the
gradients agree. PyTorch runs in float64 so the comparison is limited by the
formulas rather than by float32 rounding.
"""

import pytest
import torch

from src.engine import Value

TOL = 1e-12


def tt(x):
    """Returns a float64 leaf tensor holding ``x`` that requires grad."""

    t = torch.tensor([x], dtype=torch.double)
    t.requires_grad = True
    return t


def test_arithmetic_matches_torch():
    """Checks a long chain of every supported arithmetic operator at once."""

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()

    at, bt = tt(-4.0), tt(2.0)
    ct = at + bt
    dt = at * bt + bt**3
    ct = ct + ct + 1
    ct = ct + 1 + ct + (-at)
    dt = dt + dt * 2 + (bt + at).relu()
    dt = dt + 3 * dt + (bt - at).relu()
    et = ct - dt
    ft = et**2
    gt = ft / 2.0
    gt = gt + 10.0 / ft
    gt.backward()

    assert g.data == pytest.approx(gt.data.item(), abs=TOL)
    assert a.grad == pytest.approx(at.grad.item(), abs=TOL)
    assert b.grad == pytest.approx(bt.grad.item(), abs=TOL)


@pytest.mark.parametrize("name", ["tanh", "sigmoid", "relu", "exp"])
@pytest.mark.parametrize("x", [-2.0, -0.5, 0.5, 2.0])
def test_activations_match_torch(name, x):
    """Checks each activation's forward value and local derivative."""

    v = Value(x)
    out = getattr(v, name)()
    out.backward()

    t = tt(x)
    ref = getattr(torch, name)(t)
    ref.backward()

    assert out.data == pytest.approx(ref.item(), abs=TOL)
    assert v.grad == pytest.approx(t.grad.item(), abs=TOL)


def test_relu_derivative_at_zero_is_one():
    """Pins the current convention at the kink in relu, which is 1, not 0.

    The derivative at exactly 0 does not exist, so an implementation has to
    pick a subgradient. ``Value.relu`` branches on ``x < 0``, which puts 0 on
    the "else" side and yields 1. PyTorch and Karpathy's micrograd both pick 0.

    This test asserts what the code does today rather than what it ought to do.
    Note that the docstring on ``Value.relu`` claims 0, so the code and its
    documentation currently disagree. Switching the branch to ``x <= 0`` would
    make all three agree; until that call is made, this pins the behaviour so
    it cannot drift silently.
    """

    v = Value(0.0)
    v.relu().backward()
    assert v.grad == 1.0

    t = tt(0.0)
    torch.relu(t).backward()
    assert t.grad.item() == 0.0  # the convention we differ from


def test_reflected_operators_match_torch():
    """Checks the reflected forms, where the left operand is a plain number.

    ``__rsub__`` and ``__rtruediv__`` are the interesting ones: neither
    operation commutes, so delegating to ``__sub__`` or ``__truediv__`` would
    silently compute the expression backwards.
    """

    a = Value(3.0)
    out = (2 + a) + (2 - a) + (2 * a) + (2 / a)
    out.backward()

    at = tt(3.0)
    ref = (2 + at) + (2 - at) + (2 * at) + (2 / at)
    ref.backward()

    assert out.data == pytest.approx(ref.item(), abs=TOL)
    assert a.grad == pytest.approx(at.grad.item(), abs=TOL)


def test_sum_builtin_works_over_values():
    """Checks that ``__radd__`` makes the builtin ``sum`` usable.

    ``sum`` starts its accumulation from the plain integer 0, so without the
    reflected add the first iteration raises. ``Neuron.__call__`` relies on this.
    """

    xs = [Value(1.0), Value(2.0), Value(3.0)]
    total = sum(xs)
    total.backward()

    assert total.data == pytest.approx(6.0, abs=TOL)
    assert [x.grad for x in xs] == [1.0, 1.0, 1.0]


def test_gradient_accumulates_over_repeated_use():
    """Checks the diamond case, where one node is reached by several paths.

    ``d = (a * 2) + (a * 3)`` gives ``a`` two downstream paths. Gradients have
    to sum to 5, which is the reason the engine accumulates into ``grad``
    instead of overwriting it.
    """

    a = Value(4.0)
    d = (a * 2) + (a * 3)
    d.backward()
    assert a.grad == pytest.approx(5.0, abs=TOL)


def test_pow_rejects_non_numeric_exponent():
    """Checks that a ``Value`` exponent is refused rather than silently wrong.

    The backward rule implemented is the constant-exponent power rule. A
    variable exponent needs a second path, ``d/dn (x ** n) = x ** n * ln(x)``,
    which is not implemented.
    """

    with pytest.raises(AssertionError):
        Value(2.0) ** Value(3.0)


def test_backward_visits_each_node_once():
    """Checks the topological sort deduplicates shared subexpressions.

    ``b`` is an operand of two different nodes. If ``build_topo`` emitted it
    twice, its ``_backward`` would run twice and double the gradient it pushes
    to ``a``.
    """

    a = Value(2.0)
    b = a * a
    c = b + b
    c.backward()

    # c = 2 * a^2, so dc/da = 4a = 8
    assert a.grad == pytest.approx(8.0, abs=TOL)


def test_mlp_gradients_match_torch():
    """Checks a full forward and backward pass of the nn library against torch.

    Builds one ``tanh`` neuron by hand on both sides so the weights match, then
    compares the gradient of a squared-error loss with respect to every
    parameter. This exercises ``Value`` the way ``src/nn.py`` actually uses it.
    """

    ws = [0.3, -0.7, 0.5]
    bias = 0.1
    x = [1.0, -2.0, 3.0]
    target = 0.5

    w = [Value(wi) for wi in ws]
    b = Value(bias)
    act = sum((wi * xi for wi, xi in zip(w, x)), b)
    loss = (act.tanh() - target) ** 2
    loss.backward()

    wt = [tt(wi) for wi in ws]
    bt = tt(bias)
    act_t = bt + sum(wi * xi for wi, xi in zip(wt, x))
    loss_t = (torch.tanh(act_t) - target) ** 2
    loss_t.backward()

    assert loss.data == pytest.approx(loss_t.item(), abs=TOL)
    assert b.grad == pytest.approx(bt.grad.item(), abs=TOL)
    for wi, wti in zip(w, wt):
        assert wi.grad == pytest.approx(wti.grad.item(), abs=TOL)
