import sys
sys.setrecursionlimit(10000)

def ack(m, n):
	ans = None;
	if m == 0: 
		ans = n+1
	elif n == 0:
		ans = ack(m-1, 1)
	else:
		ans = ack(m-1, ack(m, n-1))
	return ans

import sys
try:
	loops = int(sys.argv[1])
except:
	loops = 6
for x in xrange(loops):
	for y in xrange(loops):
		ans = ack(x, y)
		print x, y, ans