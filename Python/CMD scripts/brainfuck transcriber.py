import operator
import random

def primefactors(num, minimum=2):
    factors = []
    while True:
        for i in range(int(minimum), int(num)):
            if num%i == 0:
                factors += [i]
                num /= i
                break
        else:
            break
    return factors + [num]

def repmultiple(text, old, new):
    s = text.replace(old, new)
    while s != text:
        text = s
        s = text.replace(old, new)
    return text

def nestaddloops(lengths):
    s = len(lengths)-1
    fin = ">"*s + "+"*int(lengths[0])
    for n in lengths[1:]:
        fin += "[<" + "+"*int(n)
    fin += ">-]"*s
    fin += "<"*s
    return fin

def nestsubloops(lengths):
    if len(lengths) == 1:
        return "-"*int(lengths[0])
    s = len(lengths)-1
    fin = ">"*s
    for n in lengths[:-1]:
        fin += "+"*n + "[<"
    fin += "-"*int(lengths[-1])
    fin += ">-]"*s
    fin += "<"*s
    return fin

def getshortestpossible(num, add=True):
    lengths = primefactors(num)
    f = nestsubloops
    if add: f = nestaddloops
    shortest = f(lengths)
    while len(lengths) > 1:
        lengths = [lengths[0] * lengths[1]] + lengths[2:]
        l = f(lengths)
        if len(l) < len(shortest):
            shortest = l
    return shortest

def shortestofnearby(num, size=5, add=True):
    if num < 0:
        num = -num
        add = not add
    shortest = getshortestpossible(num, add)
    s, s2 = "-", "+"
    if add: s, s2 = "+", "-"
    for i in range(1, size+1):
        newitem = getshortestpossible(num + i, add) + s2*i
        if len(newitem) < len(shortest):
            shortest = newitem
        if num - i <= 0:
            continue
        newitem = getshortestpossible(num - i, add) + s*i
        if len(newitem) < len(shortest):
            shortest = newitem
    return shortest

adds = [shortestofnearby(i, 10, True) for i in range(256)]
subs = [shortestofnearby(i, 10, False) for i in range(256)]

def travel(current, new):
    if current > new:
        return "<"*(current - new)
    return ">"*(new - current)
   
def transcribe(word):
    current = 0
    scribe = ""
    for l in word:
        v = ord(l)
        if v > current:
            scribe += adds[v - current]
        else:
            toadd = "[-]" + adds[v]
            if len(toadd) > len(subs[current - v]):
                scribe += subs[current - v]
            else:
                scribe += toadd
        scribe += "."
        current = v
    return scribe + "[-]"

def memtranscribe(text):
    cache = {}
    scribe = ""
    bank = {}
    for l in text:
        v = ord(l)
        if v in bank: bank[v] += 1
        else: bank[v] = 1
    for k, c in sorted(bank.items(), key=operator.itemgetter(1)):
        cache[k] = len(cache)
        scribe += adds[k] + ">"
    current = len(cache)
    for l in text:
        v = ord(l)
        scribe += travel(current, cache[v]) + "."
        current = cache[v]
    return repmultiple(repmultiple(scribe, "<>", ""), "><", "")

'''def bothtranscribe(text):
    current = 0
    cache = {}
    bank = {}
    scribe = "."
    for l in text:
        v = ord(l)
        if l in bank:
            bank[v] += 1
        else:
            bank[v] = 1
    for l in text:
        v = ord(l)
        possible = {}
        if k in cache:
            possible[travel(current, cache[v])] = cache[v]
        close = -1
        for k in cache:
            if bank[k] <= 0 and abs(v-k) < abs(v-close):
                close = k
        if not close == -1:
            d = close - v
            if d < 0:
                possible[travel(current, cache[close]) + "+"*(-d)] = cache[close]
            else:
                possible[travel(current, cache[close]) + "-"*d] = cache[close]
        possible[travel(current, len(cache)) + adds[v]] = len(cache)
        shortest = ""
        for p in possible:
            if shortest == []:
                shortest = p
            elif len(p) < len(shortest):
                shortest = p
        if possible[shortest] == cache[close]:
            del cache[close]
            cache[v] = close
        
        bank[v] -= 1'''

def chooseshortest(text):
    a = transcribe(text)
    b = memtranscribe(text)
    if len(a) < len(b):
        return a
    return b

def wrap(text, width=32, tab=4):
    scribe = ""
    while len(text) > width:
        scribe += " "*tab + text[:width] + "\n"
        text = text[width:]
    return scribe + " "*tab + text

ns = "this is a test message containing an economy of variety within its confines"
#print(wrap(transcribe(ns), 60))
for i, e in enumerate(adds):
    print(str(i) + ":" + str(e))

'''words = []
with open("thesaurus.txt", "r") as f:
    for line in f:
        words += [line[:-1]]

for i in range(160, 161):
    scribe = ""
    while i > 0:
        possible = [w for w in words if len(w) < i]
        if possible == []:
            break
        use = random.choice(possible) + " "
        i -= len(use) + 1
        scribe += use
    print scribe
    print transcribe(scribe)
    print memtranscribe(scribe)
    if len(memtranscribe(scribe)) > len(transcribe(scribe)):
        print "mem"
    else:
        print "normal"'''


        
