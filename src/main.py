from value import Value
from visualise import draw


def setup_function():
    a = Value(2, label='a')
    b = Value(3, label='b')
    c = Value(5, label='c')
    d = a*b
    d.label = 'd'
    e = d+c
    e.label = 'e'
    x = e.tanh()
    x.label = 'x'
    return x


def display_function(x: Value):
    dot = draw(x)
    dot.render('output', format='png', cleanup=True)


if __name__ == '__main__':
    x = setup_function()
    x.backprop()
    display_function(x)
