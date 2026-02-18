import jittor.optim as optim


class AdamW:

    def __init__(self, params, lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8, **kwargs):
        del eps, kwargs
        if hasattr(optim, 'AdamW'):
            self._opt = optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
        else:
            self._opt = optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas)

    def step(self, loss=None):
        if loss is not None:
            self._opt.step(loss)

    def zero_grad(self):
        self._opt.zero_grad()
