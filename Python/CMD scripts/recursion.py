def fibon(n):
    if n is 1 or n is 2:
        return n
    else:
        return fibon(n-1) + fibon(n-2)

def pascal(x, y):
    if y is 0 or y is x:
        return 1
    else:
        return pascal(x-1, y) + pascal(x-1, y-1)

for i in xrange(1, 20):
    print fibon(i)
print pascal(5, 4)
