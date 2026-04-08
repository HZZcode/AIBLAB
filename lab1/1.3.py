def f(expr: str, *before: str):
    g = {'s': [2, 4, 0, 1, 3, 9, 5, 8, 6, 7], 't': [2, 6, 8, 4]}
    for code in before:
        print(f">>> {code}")
    print(f">>> {expr}")
    for code in before:
        exec(code, g)
    print(eval(expr, g))

f('2 in s')
f('s + t')
f('t * 2')
f('s[1:4]')
f('s[::2]')
f('s', 's.append(10)')
f('s', 's.sort()')
f('s', 's.sort(reverse=True)')
f('s', 's.reverse()')