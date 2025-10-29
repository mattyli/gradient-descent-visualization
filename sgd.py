import numpy as np

"""
Function that executes stochastic gradient descent (SGD) on a given loss surface.
Returns the trajectory of (x, y, f(x, y)) points visited during optimization.
"""

def SGD(surface,
        x0,
        y0,
        lr,
        steps 
        ):
    
    xs = []
    ys = []
    zs = []

    x, y = x0, y0
    for t in range(steps):
        xs.append(x)
        ys.append(y)
        zs.append(surface.f(x, y))

        dx, dy = surface.grad_f(x, y)
        x = x - lr * dx
        y = y - lr * dy

    return np.array(xs), np.array(ys), np.array(zs)