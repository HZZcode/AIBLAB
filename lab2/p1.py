gcd = lambda a, b: (-gcd(-a, -b) if b < 0 else gcd(-a, abs(b)) if a < 0 else max(a, b)
    if a * b == 0 else gcd(x := min(a, b), max(a, b) - x))
reduce = lambda *nd: list(map(lambda x: x / gcd(*nd), nd))
add = lambda x, y: reduce(x[0] * y[1] + y[0] * x[1], x[1] * y[1])
sub = lambda x, y: reduce(x[0] * y[1] - y[0] * x[1], x[1] * y[1])
mul = lambda x, y: reduce(x[0] * y[0], x[1] * y[1])
div = lambda x, y: reduce(x[0] * y[1], x[1] * y[0])
s = lambda x: '/'.join(map(str, map(int, x)))
output = lambda x: print(s(x))
get_rational = lambda s: list(map(int, s.strip('()').split('/')))

"""
>>> python p1.py --% --op add --x (2/3) --y (-70/40)
-13/12
>>> python p1.py --% --op sub --x (-20/3) --y (120/470)
-976/141
>>> python p1.py --% --op mul --x (-6/19) --y (-114/18)
2/1
>>> python p1.py --% --op div --x (-6/19) --y (-114/-28)
-28/361
"""

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', choices=['add', 'sub', 'mul', 'div'])
    parser.add_argument('--x')
    parser.add_argument('--y')
    args = parser.parse_args()
    
    if args.op and args.x and args.y:
        x = get_rational(args.x)
        y = get_rational(args.y)
        ops = {'add': add, 'sub': sub, 'mul': mul, 'div': div}
        res = ops[args.op](x, y)
        output(res)