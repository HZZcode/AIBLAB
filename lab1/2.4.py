is_sorted = lambda s: all(a < b for a, b in zip(s[:-1], s[1:]))

def qsort(s):
    if len(s) <= 1: return s
    s_less = []; s_greater = []; s_equal = []
    for k in s:
        if k < s[0]:
            s_less.append(k)
        elif k > s[0]:
            s_greater.append(k)
        else:
            s_equal.append(k)
    return qsort(s_less) + s_equal + qsort(s_greater)

binary_search = lambda s, low, high, k: -1 if low > high else mid if s[mid := (low + high) // 2] == k else binary_search(s, low, mid - 1, k) if s[mid] > k else binary_search(s, mid + 1, high, k)

s = [5, 6, 21, 32, 51, 60, 67, 73, 77, 99]
if not is_sorted(s):
    s = qsort(s)
    print(s)
 
print(binary_search(s, 0, len(s) - 1, 5))
print(binary_search(s, 0, len(s) - 1, 31))
print(binary_search(s, 0, len(s) - 1, 99))
print(binary_search(s, 0, len(s) - 1, 64))
print(binary_search(s, 0, len(s) - 1, 51))