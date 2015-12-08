import random

scale = 100

chains = [[0] for i in range(1000*scale)]

for _ in range(950*scale):
    a = random.randrange(len(chains))
    b = random.randrange(len(chains))
    if a == b:
        continue
    chains[a] += chains[b]
    del chains[b]

print(len(chains))
for c in sorted(chains, key=len)[::scale]:
    print(len(c)*"*" + str(len(c)))