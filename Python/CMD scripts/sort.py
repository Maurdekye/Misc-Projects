import random

default_comparison = lambda x, y: x <= y

def randlist(size, rnge=range(100)):
    return [random.randrange(rnge[0], rnge[-1]+1) for x in xrange(size)]

def issorted(l, comparison=default_comparison):
    for i in xrange(len(l)-1):
        if not comparison(l[i], l[i+1]):
            print "Fails at index", str(i) + ", where", l[i], "is not less than", l[i+1]
            return False
    return True

# sorts

def bubblesort(l, comparison=default_comparison):
    for i in xrange(len(l)):
        for e in xrange(len(l)-1-i):
            if not comparison(l[e], l[e+1]):
                l[e], l[e+1] = l[e+1], l[e]
    return l

def cocktailsort(l, comparison=default_comparison):
    for i in xrange(len(l)):
        for e in xrange(i/2, len(l) - i/2 - 1, i%2*2-1):
            if not comparison(l[e], l[e+1]):
                l[e], l[e+1] = l[e+1], l[e]
    return l

def insertionsort(l, comparison=default_comparison):
    fin = []
    for i in l:
        for ie, e in enumerate(fin):
            if comparison(i, e):
                fin.insert(ie, i)
                break
        else:
            fin += [i]
    return fin

def mergesort(l, comparison=default_comparison):
    # base cases
    s = len(l)
    if s in [0, 1]:
        return l
    if s == 2:
        if comparison(l[1], l[0]):
            return l[::-1]
        else:
            return l
    # recursion
    split = s/2
    left = mergesort(l[:split], comparison)
    right = mergesort(l[split:], comparison)
    # merging
    fin = []
    for i in xrange(len(l)):
        if comparison(left[0], right[0]):
            fin += [left.pop(0)]
        else:
            fin += [right.pop(0)]
        if len(left) == 0:
            fin += right
            break
        elif len(right) == 0:
            fin += left
            break
    return fin

def quicksort(l, comparison=default_comparison):
    # base cases
    if len(l) in [0, 1]:
        return l
    # recursion
    left = quicksort([e for e in l[1:] if comparison(e, l[0])], comparison)
    right = quicksort([e for e in l[1:] if not comparison(e, l[0])], comparison)
    # merging
    return left + [l[0]] + right

def mixedsort_merge(l, comparison=default_comparison):
    # base cases
    s = len(l)
    if s in [0, 1]:
        return l
    if s == 2:
        if comparison(l[1], l[0]):
            return l[::-1]
        else:
            return l
    # recursion
    split = s/2
    left = mixedsort_quick(l[:split], comparison)
    right = mixedsort_quick(l[split:], comparison)
    # merging
    fin = []
    for i in xrange(len(l)):
        if comparison(left[0], right[0]):
            fin += [left.pop(0)]
        else:
            fin += [right.pop(0)]
        if len(left) == 0:
            fin += right
            break
        elif len(right) == 0:
            fin += left
            break
    return fin

def mixedsort_quick(l, comparison=default_comparison):
    # base cases
    if len(l) in [0, 1]:
        return l
    # recursion
    left = mixedsort_merge([e for e in l[1:] if comparison(e, l[0])], comparison)
    right = mixedsort_merge([e for e in l[1:] if not comparison(e, l[0])], comparison)
    # merging
    return left + [l[0]] + right

# main

if __name__ == "__main__":
    import time
    import sys
    sys.setrecursionlimit(4000)
    
    print "Generating list..."
    lst = randlist(10000, range(10000))
    print "Generated list of", len(lst), "items."

    sorts = {
        "Bubblesort" : bubblesort,
        "Insertion sort" : insertionsort,
        "Merge sort" : mergesort,
        "Quicksort" : quicksort,
        "Mixed sort" : mixedsort_merge
        }

    for sname, mysort in sorts.iteritems():
        mark1 = time.clock()
        s = mysort(lst)
        mark2 = time.clock()
        print sname, "took", mark2 - mark1, "seconds."
        if issorted(s):
            print "...list is sorted."
        else:
            print "List was not sorted!"
    
    
    
