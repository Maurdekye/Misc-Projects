def p(n): return not any([n%l == 0 for l in range(2, n//2+1)])
def pf(n):
    ps = [n for n in range(2, n) if p(n)]
    rl = []
    while not n in ps:
        for pr in ps:
            if n%pr == 0:
                n //= pr
                rl += [pr]
                break
        else:
            break
    return rl + [n]
with open("primes.txt", "w") as f:
    for i in range(1000000):
        f.write("{}: {}\n".format(i, pf(i)))
