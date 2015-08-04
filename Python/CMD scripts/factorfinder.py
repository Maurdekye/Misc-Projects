def get_factors(num):
    if num <= 1:
        return []
    factors = []
    for i in xrange(2, num/2+1):
       if num % i == 0:
           factors.append(i)
    return factors + [num]

def factor_polynomial(a, b, c):
    terms = [a, b, c]
    take_out = 1
    if terms[0] < 0:
        terms = [-t for t in terms]
        take_out = -take_out
    factors = [set(get_factors(abs(x))) for x in terms]
    shared_factors = list(factors[0] & factors[1] & factors[2])
    if len(shared_factors) > 0:
        max_factor = max(shared_factors)
        take_out = max_factor
        terms = [f/take_out for f in terms]
        factors = [get_factors(abs(x)) for x in terms]

    factors = [set(list(f) + [1]) for f in factors]

    left_pairs = [[f, terms[0]/f] for f in factors[0]]
    right_pairs = [[f, abs(terms[2])/f] for f in factors[2]]

    result = None

    if terms[2] < 0:
        for lp in left_pairs:
            for rp in right_pairs:
                if (lp[0]*rp[1]) - (lp[1]*rp[0]) == terms[1]:
                    result = [lp[0], -rp[0], lp[1], rp[1]]
    else:
        for lp in left_pairs:
            for rp in right_pairs:
                prospect = (lp[0]*rp[1]) + (lp[1]*rp[0])
                if prospect == terms[1]:
                    result = [lp[0], rp[0], lp[1], rp[1]]
                elif prospect == -terms[1]:
                    result = [lp[0], -rp[0], lp[1], -rp[1]]

    if result == None: return None
    return result + [take_out]

def pretty_factored_polynomial(result):
    if result == None:
        return None

    to_print = ""
    if result[4] < 0:
        to_print += "-"
    if abs(result[4]) != 1:
        to_print += str(abs(result[4]))
    to_print += "("
    if result[0] != 1:
        to_print += str(result[0]) + "x"
    else:
        to_print += "x"
    if result[1] < 0:
        to_print += " - "
    else:
        to_print += " + "
    to_print += str(abs(result[1]))
    to_print += ")("
    if result[2] != 1:
        to_print += str(result[2]) + "x"
    else:
        to_print += "x"
    if result[3] < 0:
        to_print += " - "
    else:
        to_print += " + "
    to_print += str(abs(result[3]))
    to_print += ")"

    return to_print

def pretty_polynomial(terms):
    polynomialprint = ""
    if terms[0] > 1 or terms[0] < -1:
        polynomialprint = str(terms[0])
    if terms[0] == -1:
        polynomialprint += "-"
    polynomialprint += "x^2 "
    if terms[1] > 1:
        polynomialprint += "+ " + str(terms[1]) + "x "
    elif terms[1] == 1:
        polynomialprint += "+ x "
    elif terms[1] == -1:
        polynomialprint += "- x "
    elif terms[1] < -1:
        polynomialprint += "- " + str(abs(terms[1])) + "x "
    if terms[2] > 0:
        polynomialprint += "+ " + str(terms[2])
    elif terms[2] < 0:
        polynomialprint += "- " + str(abs(terms[2]))

    return polynomialprint

if __name__ == "__main__":
    import re
    inp = raw_input("What are the terms of the polynomial? ")
    termstrings = re.split("\s+", inp.strip())[:3]
    terms = []
    for t in termstrings:
        try:
            terms.append(int(t))
        except ValueError:
            raw_input(t + " is not a number.\n\n")
            exit()
    if terms[0] == 0:
        raw_input("The first term must not be 0.\n\n")
        exit()
        
    factored = factor_polynomial(*terms)
    pretty_factored = pretty_factored_polynomial(factored)
    pretty_poly = pretty_polynomial(terms)
    
    if factored == None:
        raw_input("There is no factor for the polynomial " + pretty_poly + "\n\n")
    else:
        raw_input("The factor for the polynomial " + pretty_poly + " is " + pretty_factored + ".\n\n")
