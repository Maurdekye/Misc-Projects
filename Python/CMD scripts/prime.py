def isprime(num, pls=[]):
    if pls == []:
        pls = list(range(2,int((num/2)+1)))
    else:
        pls = [x for x in pls if x < (num/2)+1]
    for i in pls:
        if num%i == 0:
            return False
    return True

import datetime

start = datetime.second()
primes = filter(isprime, list(range(1000)))

print(primes)
lap = datetime.second()
print("Mark 1: " + (lap - start))
print(filter(isprime, zip(list(range(1000)))), primes, end=" "))
print("Mark 2: " + (datetime.second() - lap))
