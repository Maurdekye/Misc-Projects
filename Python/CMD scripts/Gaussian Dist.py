from random import randint as rnd

die = 40

def roll(): return sum([rnd(1,6) for i in xrange(die)])

rolls = {i : 0 for i in xrange(die, (die*6)+1)}

for i in xrange(100000): rolls[roll()] += 1

for k in sorted(rolls): print str(k) + ": " + "X" * int(rolls[k]/80)

raw_input()
