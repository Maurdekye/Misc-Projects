from random import randrange as rnd
import time

# Sorting Comparisons

alphabet = [
    '.', ',', '-', '!', '?', '|',
    '0', '1', '2', '3', '4',
    '5', '6', '7', '8', '9',
    ' ', 'a', 'b', 'c', 'd', 'e',
    'f', 'g', 'h', 'j', 'k',
    'l', 'm', 'n', 'o', 'p',
    'q', 'r', 's', 't', 'u',
    'v', 'w', 'x', 'y', 'z']

def fixto(word, alph=alphabet):
    new = ''
    for letter in word:
        if letter.upper() in alph: new += letter.upper()
        if letter.lower() in alph: new += letter.lower()
    return new

def word_ahead(word, compare, alph=alphabet):
    word, compare = fixto(word, alph), fixto(compare, alph)
    if word == '': return True
    if compare == '': return False
    if word[0] == compare[0]: return word_ahead(word[1:], compare[1:])
    for letter in alph:
          if word[0] == letter: return True
          if compare[0] == letter: return False
    return True

def reverse(a, b): return a < b

# List applications

def list_sorted(ls, comp=lambda a, b: a > b):
    for i in xrange(len(ls)-1):
        if comp(ls[i], ls[i+1]): return False
    else: return True

def randls(size=20, maxval=100): return [rnd(maxval)+1 for i in xrange(size)]

def empty(ls): return (len(ls) > 0)

def mkls(item):
    if type(item) != list:
        return [item]
    return item

# Sorting algorithms

def insertionsort(ls, comp=lambda a, b: a > b):
    to_sender = []
    for i in ls:
        for j in xrange(len(to_sender)):
            if comp(to_sender[j], i):
                to_sender.insert(j, i)
                break
        else: to_sender.append(i)
    return to_sender

def quicksort(ls, comp=lambda a, b: a > b):
    if ls == []: return ls
    smaller = quicksort([x for x in ls[1:] if not comp(x, ls[0])])
    bigger = quicksort([x for x in ls[1:] if comp(x, ls[0])])
    return smaller + [ls[0]] + bigger

def bubblesort(ls, comp=lambda a, b: a > b):
    for a in ls:
        noswitch = True
        for j in xrange(len(ls)-1):
            if comp(ls[j], ls[j+1]):
                noswitch = False
                ls[j], ls[j+1] = ls[j+1], ls[j]
        if noswitch: break
    return ls

def betterbubblesort(ls, comp=lambda a, b: a > b):
    fin = []
    for a in ls:
        noswitch = True
        for j in xrange(len(ls)-1):
            if comp(ls[j], ls[j+1]):
                noswitch = False
                ls[j], ls[j+1] = ls[j+1], ls[j]
        if noswitch: break
        if ls != []: fin = [ls.pop()] + fin
    return fin

def pivotsort(ls, comp=lambda a, b: a > b): # Broken
    if ls == []: return []
    if len(ls) == 2:
        if not list_sorted(ls, comp):
            return ls[1] + ls[0]
        else: return ls
    if not list_sorted(ls, comp):
        piv = int(len(ls)/2)
        ls = pivotsort(ls[piv:]) + pivotsort(ls[:piv])
    return ls

def mergesort(ls, comp=lambda a, b: a > b):
    if len(ls) in [0, 1]: return ls
    hlen = len(ls) / 2
    print ls
    a, b = mergesort(ls[:hlen]), mergesort(ls[hlen:])
    print a, b
    new = []
    while a != [] and b != []:
        if comp(a[-1], b[-1]):
            new += [a.pop()]
        else: new += [b.pop()]
    new += a + b
    return new

def bogsort(ls):
    count = 0
    while not list_sorted(ls):
        count += 1
        switch = (rnd(0, len(ls)-1), rnd(0, len(ls)-1))
        ls[switch[0]], ls[switch[1]] = ls[switch[1]], ls[switch[0]]
    print count
    return ls

sorts = [
    quicksort,
    bubblesort
    ]
testmag = 400

avgs = []
for sort in sorts:
    avg = 0.0
    for i in xrange(testmag):
        if abs(i%(testmag / 10)) < 2: print 'Sort #' + str(i)
        tosort = randls(size=testmag, maxval=testmag)
        start = time.clock()
        sort(tosort)
        avg += time.clock() - start
    print avg
