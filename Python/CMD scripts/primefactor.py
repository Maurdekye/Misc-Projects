import random

def tprime(n):
  for i in range(2, int(n/2) - 1):
    if n % i == 0:
      return False
  return True

plist = [n for n in range(2, 1000) if tprime(n)]

def factors(n):
  facts = []
  while True:
    for p in plist:
      if p >= n:
        return facts + [int(n)]
      if n % p == 0:
        facts.append(p)
        n /= p
        break
  return facts + [int(n)]

cops = 1
cofs = 1

i = 0

try:
  while True:
    af = factors(random.randrange(2, 1000))
    bf = factors(random.randrange(2, 1000))
    if bool(set(af) & set(bf)):
      cops += 1
    else:
      cofs += 1
    i+=1
    if i%10000 == 0:
      print(cops / cofs)
except KeyboardInterrupt:
  print("Program stopped")
  print("Coprimes vs cofactors:", cops, cofs)
  print("generated numbers:", i)