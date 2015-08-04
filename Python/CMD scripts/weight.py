from random import randint as rnd
import math

def dist(point1, point2):
    const1 = point2[0] - point1[0]
    const2 = point2[1] - point1[1]
    return (const1**2 + const2**2)**0.5


def weighted(points):
    to_sender = {x : 0 for x in points}
    for p in points:
        for cp in points:
            if p != cp:
                to_sender[p] += 1/dist(p, cp)
    return to_sender

def mPrint(mat):
    for row in mat:
        for item in row:
            if item == 0.0: print "   ",
            else: print item,
        print

matte = [[0.0 for x in xrange(35)] for y in xrange(35)]
points = [(rnd(1,34), rnd(1,34)) for x in xrange(100)]
weights = weighted(points)
for p in weights:
    matte[p[0]][p[1]] = round(weights[p], 1)
mPrint(matte)
