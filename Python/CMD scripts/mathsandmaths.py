import math

def cycle(lis, amount):
    amount = amount % len(lis)
    return lis[amount:] + lis[:amount]

def similarity_pattern_check(lis, depth=5, minimum=2):
    depth = max(depth, len(lis))
    minimum = max(depth, minimum)
    for size in range(minimum, depth-1)[::-1]:
        for c in xrange(size):
            check = cycle(lis[-size:], c)
            for i, k in enumerate(lis[-depth:-size-1]):
                if lis[i:i+size] == check: return True
    return False

def khintche(a, recurs=50):
    if recurs <= 0: return a
    return a + (1.0 / seq(a, recurs-1))
                           
def fiblist(n):
    if n <= 0: return []
    if n == 1: return [1]
    ls = [1, 1]
    for i in xrange(n-2):
        ls += [ls[-1] + ls[-2]]
    return ls

def something_a(n):
    vals = []
    while True:
        vals += [n]
        if n%2 == 0: n /= 2
        else: n += 1
        print n,
        if similarity_pattern_check(vals, 5, 1):
            print
            return vals + [n]

l = [1, 2, 1, 2, 1, 2, 1]
print similarity_pattern_check()
#print something_a(60)
