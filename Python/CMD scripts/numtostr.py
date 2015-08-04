numtable = {e : i for i, e in enumerate(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])}
def tonum(nstr):
    decimal, num = 0, 0
    if "." in nstr:
        nstr, dstr = nstr.split(".", 1)
        for i, l in enumerate(dstr):
            decimal += numtable[l]*10**(-i)
    for i, l in enumerate(list(reversed(nstr))):
        num += numtable[l]*10**i
    return num + decimal
