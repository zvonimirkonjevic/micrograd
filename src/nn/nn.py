class Module:
    """Base class for anything that owns trainable parameters.

    Subclasses override :meth:`parameters` to expose their own ``Value``
    objects; the gradient bookkeeping is then inherited for free.
    """


    def zero_grad(self):
        """Resets the gradient of every parameter to zero.

        Gradients accumulate across backward passes, so this must be called
        before each optimization step or gradients from previous steps leak
        into the current one.
        """

        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        """Returns the trainable parameters of this module.

        Returns:
            An empty list. Subclasses override this to return their ``Value``
            parameters.
        """

        return []