from functools import total_ordering
from p1 import *

@total_ordering
class Rational:
    v = property(lambda l: l[0] / l[1])
    def __init__(self, *l): self._l = l
    def __getitem__(self, key): return self._l[key]
    def __add__(self, other): return Rational(*add(self, other))
    def __sub__(self, other): return Rational(*sub(self, other))
    def __mul__(self, other): return Rational(*mul(self, other))
    def __truediv__(self, other): return Rational(*div(self, other))
    def __eq__(self, other): return (self - other)[0] == 0
    def __lt__(self, other): return (self - other)[0] < 0
    def __str__(self): return s(self)

def test():
    testsuite = [
        ('Rational(2, 3) + Rational(-70, 40)', Rational(-13, 12)),
        ('Rational(-20, 3) - Rational(120, 470)', Rational(-976,141)),
        ('Rational(-6, 19) * Rational(-114, 18)', Rational(2, 1)),
        ('Rational(-6, 19) / Rational(-114, -28)', Rational(-28,361)),
        ('Rational(-6, 19) == Rational(-14, 41)', False),
        ('Rational(-6, 19) != Rational(-14, 41)', True),
        ('Rational(6, -19) > Rational(14, -41)', True),
        ('Rational(-6, 19) < Rational(-14, 41)', False),
        ('Rational(-6, 19) >= Rational(-14, 41)', True),
        ('Rational(6, -19) <= Rational(14, -41)', False),
        ('Rational(-15, 8) == Rational(120, -64)', True),
    ]
    for t in testsuite:
        try:
            result = eval(t[0])
        except:
            print('Error in evaluating ' + t[0]); continue
        if result != t[1]:
            print('Error:  %s != %s' % (t[0], t[1]))

if __name__ == '__main__':
    test()