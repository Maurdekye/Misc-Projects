import time
from random import randrange as rr

rn = lambda: rr(-20, 20)

dot1 = lambda a, b: sum([m * n for m, n in zip(a, b)])
dot2 = lambda a, b: a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

print("generating values")
l = [(rn(), rn(), rn()) for x in range(2000000)]
f = []
print("running test on dot1")
start = time.clock()
for i, a in enumerate(l[:-1]):
    b = l[i+1]
    f.append(dot1(a, b))
diff = time.clock() - start
print("took {} seconds".format(diff))
print("running test on dot2")
start = time.clock()
for i, a in enumerate(l[:-1]):
    b = l[i+1]
    f.append(dot2(a, b))
diff = time.clock() - start
print("took {} seconds".format(diff))
