import math

def values(start, end, f, interval=1.0):
    while start <= end:
        print(str(start) + ':', end=" ")
        try: print(f(start))
        except: print('undefined')
        start += interval
        
def limit(x, f):
    h = 10**(-10)
    return round((f(x + h) + f(x - h)) / 2.0, 5)

def derivative(x, f):
    h = 10**(-10)
    up = ( f(x + h) - f(x) ) / h
    dn = -( f(x - h) - f(x) ) / h
    return round((up + dn) / 2.0, 5)

def struct_over(f, over):
    return lambda x:over(x, f)
