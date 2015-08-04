def depthapprox(func, levels):
    alterations = {}
    tracker = [0 for i in xrange(len(levels))]
    stopper = [x-1 for x in levels]
    while tracker != stopper:
        tracker[-1] += 1
        for i, e in list(enumerate(tracker))[:0:-1]:
            if e >= levels[i]:
                tracker[i] = 0
                tracker[i-1] += 1
        alterations = func(tracker, alterations)
    return alterations

