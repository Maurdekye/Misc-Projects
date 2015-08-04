import random
import time

def printg(ingrid):
    for row in ingrid:
        for item in row:
            print(item, end="")
        print()
    print()

def randgrid(width=3, height=None, range=(10, 100)):
    if height is None: height = width
    return [[random.randrange(range[0], range[1]) for x in xrange(width)] for y in xrange(height)]

def identity(size=3):
    return [[1 if x == y else 0 for x in xrange(size)] for y in xrange(size)]

def isRect(ingrid):
    l = None
    for r in ingrid:
        if l == None: l = len(r)
        elif len(r) != l: return False
    return True

def isEven(ingrid):
    return isRect(ingrid) and len(ingrid[0]) == len(ingrid)

def determinate(ingrid ,debug=False):
    def deb(msg):
        if debug: print(msg)
    if not isEven(ingrid): return None
    glen = len(ingrid)

    a = 0
    for x in xrange(glen):
        rsum = 1
        for y in xrange(glen):
            rsum *= ingrid[y%glen][(y+x)%glen]
        deb(rsum)
        a += rsum
    deb('\t' + str(a) + '\n')

    b = 0
    for x in xrange(glen):
        rsum = 1
        for z, y in zip(xrange(glen), xrange(glen-1, -1, -1)):
            rsum *= ingrid[z%glen][(y+x)%glen]
        deb(rsum)
        b += rsum
    deb('\t' + str(b) + '\n')

    return a - b

def inverse(ingrid):
    det = determinate(ingrid)
    if det is None or det == 0: return None
    
grid = [
    [2, 3, 5],
    [7, 9, 11],
    [10, 1, 4]
    ]
for i in xrange(1000, 1010, 10) :
    print(i)
    grid = randgrid(i, range=(10, 100))
    #printg(grid)
    print("Calculating determinate of {0}x{1} grid...".format(len(grid[0]), len(grid)))
    start = time.clock()
    print(determinate(grid))
    end = time.clock()
    print("Finished; took {0} seconds.".format(end - start))
