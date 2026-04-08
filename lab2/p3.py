from math import sqrt, cos, sin, exp
from dataclasses import dataclass

@dataclass
class Integrator:
    a: int; b: int; n: int
    def compute_points(self, *_): ...
    def integrate(self, f): return sum(map(lambda t: t[1] * f(t[0]), zip(*self.compute_points(
        h := (self.b - self.a) / self.n, [self.a + h * i for i in range(self.n + 1)])))) # type: ignore

[globals().__setitem__(s, type(s, (Integrator,), {'compute_points': f})) for s, f in {
    'Trapezoidal': lambda s, h, x: (x, [h / ((i in [0, s.n]) + 1) for i in range(s.n + 1)]),
    'Simpson': lambda s, h, x: (x, [(((1 + i % 2 * 2) * (i not in [0, s.n]) + 1) * h / 3) for i in range(s.n + 1)]),
    'GaussLegendre': lambda s, h, x: zip(*[[0.5 * h * k / sqrt(3) + (x[i] + x[i + 1]) / 2, 0.5 * h] for i in range(s.n) for k in [-1, 1]])
}.items()]

def test():
    def f(x): return (x * cos(x) + sin(x)) * exp(x * sin(x))
    def F(x): return exp(x * sin(x))

    a = 2; b = 3; n = 200
    I_exact = F(b) - F(a)
    tol = 1E-3

    methods = [Trapezoidal, Simpson, GaussLegendre] # type: ignore
    for method in methods:
        integrator = method(a, b, n)
        I = integrator.integrate(f)
        rel_err = abs((I_exact - I) / I_exact)
        print('%s: %g' % (method.__name__, rel_err))
        if rel_err > tol:
            print('Error in %s' % method.__name__)

if __name__ == '__main__':
    test()