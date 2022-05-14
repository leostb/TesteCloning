import numpy as np
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
import matplotlib.pyplot as plt


def gold(x, y):
    a = 1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)
    b = 30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2)
    return a * b


def suums(x, y):
    return x ** 2 + 2 * y ** 2


def dejong(x, y):
    return 100 * (x**2 - y**2)**2 + (1 - x**2)**2


def ackley(x, y):
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x ** 2 + y ** 2))) - exp(0.5 * (cos(2 *
                                                                              pi * x) + cos(2 * pi * y))) + e + 20


@np.vectorize
def bump(x, y):
    if x * y < 0.75:
        return float("NaN")
    elif x + y > 14:
        return float("NaN")
    temp0 = cos(x) ** 4 + cos(y) ** 4
    temp1 = 2 * cos(x) ** 2 * cos(y) ** 2
    temp2 = sqrt(x ** 2 + 2 * y ** 2)
    return -abs((temp0 - temp1) / temp2)


def rastrigin(x, y):
    z = x ** 2 + y ** 2 - 10 * cos(2 * pi * x) - 10 * cos(2 * pi * y) + 10
    return -z

def printgraph(r_min, r_max, density, f):
    x = np.linspace(r_min, r_max, density)
    y = np.linspace(r_min, r_max, density)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='jet', shade='false')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(25, 225)
    plt.show()


printgraph(-2, 2, 100, gold)
printgraph(-10, 10, 100, suums)
printgraph(-2, 2, 100, dejong)
printgraph(-32.768, 32.768, 32, ackley)
printgraph(-5, 5, 100, rastrigin)
