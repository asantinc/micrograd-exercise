import numpy as np

from visualise import draw


class Value:
    def __init__(self, val, children=(), op='', label='') -> None:
        self.data = val
        self._prev = set(children)
        self._op = op
        self.grad = 0
        self.label = label
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f'Value({self.data}) {self._op}'

    def __add__(self, other) -> "Value":
        out = Value(self.data + other.data, children=(self, other), op='+')

        def _backward():
            self.grad += 1*out.grad
            other.grad += 1*out.grad
        self._backward = _backward
        return out

    def __mul__(self, other) -> "Value":
        out = Value(self.data * other.data, children=(self, other), op='*')

        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        self._backward = _backward
        return out

    def tanh(self):
        def _backward():
            self.grad += (1 - np.tanh(self.data)**2)*out.grad
        self._backward = _backward
        out = Value(np.tanh(self.data), children=(self,), op='tanh')
        return out


a = Value(2, label='a')
b = Value(3, label='b')
c = Value(5, label='c')
d = a*b
d.label = 'd'
e = d+c
e.label = 'e'
x = e.tanh()
x.label = 'x'


dot = draw(x)
dot.render('output', format='png', cleanup=True)
