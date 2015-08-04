import math

def values(start, end, f, interval=1.0):
    while start <= end:
        print(str(start) + ':', end="")
        try: print(f(start))
        except: print('undefined')
        start += interval
        
def limit(x, f):
    h = 10**(-10)
    return round((f(x + h) + f(x - h)) / 2.0, 5)

def derivative(f):
    def ret(x):
        h = 10**(-10)
        up = ( f(x + h) - f(x) ) / h
        dn = -( f(x - h) - f(x) ) / h
        return round((up + dn) / 2.0, 5)
    return ret

def definite_integral(f, a, b, p=100):
    rng = range(1, p+1)
    d = (b - a) / p
    return sum( [f(d*n + a)*d for n in rng] ) + ( d*sum( f(d*(n-1) + a) - f(d*n + a) ) ) / 2.0

def indefinite_integral(f, c, p=100):
    return lambda x: definite_integral(f, c, x, p)

def fast_indef_integral(f, values, p=10, negative=False):
    subvalues = []
    for i, v in enumerate(values[1:]):
        subvalues.append(indefinite_integral(f, values[i-1], v, p))
    s = 0
    integrals = []
    for i in subvalues:
        integrals += [s]
        s += i
    return integrals


def struct_over(f, over):
    return lambda x:over(x, f)