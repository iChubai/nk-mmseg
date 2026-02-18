"""Base transform class."""


class BaseTransform:

    def __call__(self, results):
        return self.transform(results)

    def transform(self, results):
        raise NotImplementedError
