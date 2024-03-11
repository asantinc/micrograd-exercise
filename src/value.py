from typing import List
import numpy as np


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
        out = Value(np.tanh(self.data), children=(self,), op='tanh')

        def _backward():
            self.grad += (1 - np.tanh(self.data)**2)*out.grad
        out._backward = _backward
        return out

    def backprop(self):
        topo: List[Value] = []
        visited = set()

        def build_topo(root) -> list:
            if root not in visited:
                visited.add(root)
                root.grad = 0.0  # reset all grads - about to re-calculate them
                for child in root._prev:
                    build_topo(child)
                topo.append(root)

        build_topo(self)
        self.grad = 1.0  # root should start with grad 1.0, else 0.0
        for node in reversed(topo):
            node._backward()
