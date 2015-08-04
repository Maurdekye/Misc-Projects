def outcall(ls, ind, invalid=0):
    try: return ls[ind]
    except IndexError: return invalid

def addtab(instr, tabspace):
    return str(instr) + ' '*max(0, tabspace - len(str(instr)))

def matrixop(mat1, mat2, op):
    fin = [[None for x in xrange(len(mat1))] for y in xrange(len(mat2))]
    for x in xrange(len(fin)):
        for y in xrange(len(fin[x])):
            fin[x][y] = op(outcall(mat1, x), outcall(mat2, y))
    return fin

def matprint(mat):
    for row in mat:
        for item in row: print addtab(item, 3),
        print
    print


matA = [-1, 2, 3]
matB = [-1, 0, 4, 9]

res = matrixop(matA, matB, lambda a, b: a - b)

print(matA)
print(matB)
matprint(res)
