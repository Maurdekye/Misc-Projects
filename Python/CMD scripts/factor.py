import math

def quadratic_formula(a, b, c):
    start = b**2 - 4*a*c
    if start < 0: return []
    inner = math.sqrt(start)
    return [(-b + inner)/(2*a),
            (-b - inner)/(2*a)]

def getpn(n, by_self=False):
    if n%1 == 0: n = int(n)
    if by_self: return str(n)
    if n >= 0: return "+ " + str(n)
    else: return "- " + str(n*(-1))

def getquad(a, b, c):
    if a == 0:
        return "Not quadratic"
    eq = ""
    if a != 1:
        eq += getpn(a, True)
    eq += "x^2 "
    if b != 0:
        if b != 1:
            eq += getpn(b)
        eq += "x "
    if c != 0:
        eq += getpn(c)
    return eq

def complete_the_square(a, b, c):
    a, b, c = float(a), float(b), float(c)
    multi = False
    if a != 1:
        multi = True

    n1 = b/(2*a)
    n2 = (c/a) - n1**2
    if multi:
        return "{a}((x {b})^2 {c})".format(a=getpn(a, True), b=getpn(n1), c=getpn(n2))
    else:
        return "(x {b})^2 {c}".format(b=getpn(n1), c=getpn(n2))

if __name__ == "__main__":
    print complete_the_square(2, 7, 15)
    
