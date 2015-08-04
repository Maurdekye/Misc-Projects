class baseNum:
    alph = [
        'A', 'B', 'C', 'D', 'E', 'F',
        'G', 'H', 'I', 'J', 'K', 'L',
        'M', 'N', 'O', 'P', 'Q', 'R',
        'S', 'T', 'U', 'V', 'W', 'X',
        'Y', 'Z']
    
    def __init__(this, value, base):
        this.value = value
        this.base = min(base, 36)
    
    def __str__(this):
        ret, large = [], this.base
        val = this.value
        while large <= val:
            large *= this.base
        while large > this.base:
            large /= this.base
            ret += [val // large]
            val %= large
        ret += [val]
        retstr = ""
        for l in ret:
            if l >= 10: retstr += str(this.alph[l-10])
            else: retstr += str(l)
        return retstr

for x in xrange(1025): print baseNum(x, 16), x
