from typing import List
import numpy as np

from value import Value
from visualise import display_function


class Neuron:
    def __init__(self, input_size):
        self._w: List[Value] = [Value(np.random.uniform(-1, 1)) for x in range(input_size)]
        for i, w_i in enumerate(self._w):
            w_i.label = f'w_{i}'
        
        self._b: Value = Value(1.0); self._b.label = 'b'
        self.activation: Value = Value(0.0)  # calculated when Neuron is called
        self.activation.label = 'Y'

    def __call__(self, x: List[Value]):
        assert len(x) == len(self._w)  # input size must match size of Neuron
        self._x = x  # the input vector
        pre_activation = sum((x*w for (x, w) in zip(self._x, self._w)), self._b)
        self.activation = pre_activation.tanh()
        print(self)

    def backward(self):
        self.activation.backprop()


class Layer:
    def __init__(input_number: int, output_number: int):
        ...

    def forward():
        ...

    def backward():
        ...


class MLP:  # multi-layer perceptron
    def __init__(input_number: int, output_numbers: List[int]):
        ...

    def forward():
        ...

    def backward():
        ...


class InputVector:
    def __init__(self, x: List[float]) -> None:
        self.x = [Value(x_i) for x_i in x]
        for i, x_i in enumerate(self.x):
            x_i.label = f'x_{i}'


if __name__ == '__main__':
    n = Neuron(3)
    input_x = InputVector([4.0, 5.0, 6.0])
    n(input_x.x)
    n.backward()
    display_function(n.activation)
