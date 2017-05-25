import random
import sys

maxnum = int(sys.argv[1])

generated = list(range(maxnum))

lvals = []
largest = 0

while len(generated) > 0:
  c = 0
  gn = random.randrange(maxnum)
  while gn not in generated:
    c += 1
    gn = random.randrange(maxnum)
  generated.remove(gn)
  if c > largest:
    largest = c
    lvals.append((c, len(generated)))
  if len(generated) < maxnum / 10:
    print("generated {} after {} attempts; {} numbers left to find".format(gn, c, len(generated)))

for cnt, ngen in lvals:
  print("Had {} tries first at {}".format(cnt, ngen))