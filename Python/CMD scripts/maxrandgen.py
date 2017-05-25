import random
import sys

maxrand = int(sys.argv[1])

c = 0
target = random.randrange(maxrand)
print("searching for target")
while random.randrange(maxrand) != target:
  c += 1

print("found after {} attempts".format(c))