LETTERS = [
    'A', 'B', 'C', 'D', 'E', 'F',
    'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X',
    'Y', 'Z']

def generate(length, letterlist=LETTERS):
    def lexify(num, by, leng=0):
        ret, large = [], by
        while large <= num: large *= by
        while large > by:
            large /= by
            ret = [num // large] + ret
            num %= large
        return [num] + ret + ([0]*max(0, leng - len(ret) - 1))
    ret, count = [""], 0
    while len(ret[-1]) <= length:
        cur = ""
        for e in lexify(count, len(letterlist), length):
            cur += letterlist[e]
        ret += [cur[::-1]]
        count += 1
    return ret[1:-1]

def form(word, letterlist=LETTERS):
    ret = ""
    for l in word:
        if l.upper() in letterlist:
            ret += l.upper()
        elif l.lower() in letterlist:
            ret += l.lower()
    return ret

def findWord(word, letterlist=LETTERS):
    def lexify(num, by, leng=0):
        ret, large = [], by
        while large <= num: large *= by
        while large > by:
            large /= by
            ret = [num // large] + ret
            num %= large
        return [num] + ret + ([0]*max(0, leng - len(ret) - 1))
    word = form(word, letterlist)
    last, count, length = "", 0, len(word)
    while len(last) <= length:
        cur = ""
        for e in lexify(count, len(letterlist), length):
            cur += letterlist[e]
        last = cur[::-1]
        #print last
        if last == word:
            return count
        count += 1
    return -1
    
def allCodes(length, letterlist=LETTERS, ret=[]):
    if length <= 0: return ret
    ret += generate(length, letterlist)
    return allCodes(length - 1, letterlist, ret)

search = "breadbox"

print findWord(search)
