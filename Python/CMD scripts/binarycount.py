counter = [False for i in xrange(10)]
while True:
    for bit in counter[::-1]:
        if bit: print '1',
        else: print '0',
    print
    raw_input()
    
    for i, bit in enumerate(counter):
        if bit:
            counter[i] = False
            break
        else:
            counter[i] = True
