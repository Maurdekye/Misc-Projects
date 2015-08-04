import math

def getpower(x, base=2):
    if x < 1:
        return 0
    else:
        return int(math.floor(math.log(base, 2)))

def tobinary(x):
    fin = 0
    while x > 0:
        p = getpower(x, 2)
        x -= 2**p
        fin += 10**p
    return fin

def frombinary(x):
    fin = 0
    s = str(x)
    for i, c in enumerate(s):
        if c == "1":
            fin += 2**i
    return fin

for x in xrange(32):
    y = tobinary(x)
    print x, y, frombinary(x)

