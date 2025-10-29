import numpy as np
import abc

# define a base class for loss surfaces, all others will inherit from it
class LossSurface(abc.ABC):
    @abc.abstractmethod
    def f(self, x, y):
        """Return scalar loss f(x, y). Supports numpy arrays."""
        pass

    @abc.abstractmethod
    def grad_f(self, x, y):
        """Return gradient [df/dx, df/dy] at (x, y). Should broadcast over arrays."""
        pass

    def default_range(self):
        """
        Return plotting ranges for x and y as (xmin, xmax, ymin, ymax).
        Subclasses can override this to give a 'nice' view.
        """
        return (-3, 3, -3, 3)

    def make_grid(self, n=100):
        """
        Utility: build meshgrid + Z for plotting surfaces.
        Uses default_range() unless you want to override.
        """
        xmin, xmax, ymin, ymax = self.default_range()
        xs = np.linspace(xmin, xmax, n)
        ys = np.linspace(ymin, ymax, n)
        X, Y = np.meshgrid(xs, ys)
        Z = self.f(X, Y)
        return X, Y, Z

# quadratic surface: f(x, y) = x^2 + y^2
class QuadraticSurface(LossSurface):
    def f(self, x, y):
        return x**2 + y**2

    def grad_f(self, x, y):
        dx = 2 * x
        dy = 2 * y
        return dx, dy

    def default_range(self):
        return (-5, 5, -5, 5)

# rosenbrock surface: f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
class Rosenbrock(LossSurface):
    def __init__(self, a=1.0, b=100.0):
        self.a = a
        self.b = b

    def f(self, x, y):
        a, b = self.a, self.b
        return (a - x)**2 + b * (y - x**2)**2

    def grad_f(self, x, y):
        a, b = self.a, self.b
        dfdx = -2*(a - x) - 4*b*x*(y - x**2)
        dfdy =  2*b*(y - x**2)
        return dfdx, dfdy

    def default_range(self):
        return (-2, 2, -1, 3)
