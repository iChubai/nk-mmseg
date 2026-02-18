import jittor.optim as optim


class SGD:

    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False, **kwargs):
        del nesterov, kwargs
        self._opt = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    def step(self, loss=None):
        if loss is not None:
            self._opt.step(loss)

    def zero_grad(self):
        self._opt.zero_grad()
