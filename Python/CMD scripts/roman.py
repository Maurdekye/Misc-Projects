keys = {
    1 : "I",
    5 : "V",
    10 : "X",
    50 : "L",
    100 : "C",
    500 : "D",
    1000 : "M"
}

order = sorted([i for i, v in keys.iteritems()])[::-1]

def getNumeral(num):
    numeral = ""
    for i, base in enumerate(order):
        z = 0
        while num >= base:
            num -= base
            numeral += keys[base]
            z += 1
        if z == 4:
            numeral = numeral[:-4] + keys[base] + keys[order[i-1]]
            if len(numeral) > 2 and numeral[-3] == numeral[-1]:
                place = numeral[-2]
                numeral = numeral[:-3] + place + keys[order[i-2]]
        if num == 0:
            break
    return numeral

for i in xrange(100):
    print getNumeral(i)