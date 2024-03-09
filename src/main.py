import numpy as np
import matplotlib.pyplot as plt

from visualise import draw


class Value:
    def __init__(self, val, children=(), op='') -> None:
        self.data = val
        self._prev = set(children)
        self._op = op

    
    def __repr__(self) -> str:
        return f'Value({self.data}) {self._op}'

    # def __radd__(self, other):
    #     return Value(self.data + other, children=(self, other), op='+')

    def __add__(self, other):
        return Value(self.data + other.data, children=(self, other), op='+')
    
    def __mul__(self, other):
        return Value(self.data * other.data, children=(self, other), op='*')
    
    # def __rmul__(self, other):
    #     return Value(self.data * other, children=(self, other), op='*')




def function(x: Value, y: Value) -> Value:
    return Value(2)*x + Value(3)*y


dot = draw(function(Value(3), Value(4)))
dot.render('output', format='png', cleanup=True)