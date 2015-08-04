def isprime(num):
    if num <= 1: return False
    if num == 4: return False
    for i in xrange(2, num/2):
        if num%i == 0:
            return False
    return True

def primefactors(num, primelist):
    to_sender = []
    if num in primelist: return num
    for prime in primelist:
        if num%prime == 0:
            num /= prime
            to_sender += [prime]
    return to_sender

primelist = [n for n in xrange(1000) if isprime(n)]
print('finished counting primes')
print 75361, primefactors(75361, primelist)
