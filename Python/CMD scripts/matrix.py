def blank_mat(width=10, height=10, typemat="-"): return [[typemat for item in range(width)] for row in range(height)]

def print_mat(mat):
    for row in mat:
        for item in row:
            print item,
        print ""
    print ""

def print_mat_condensed(mat):
    for row in mat:
        s = ""
        for item in row:
            s += str(item)
        print s
    print ""

def print_mat_numbered(mat):
    s = ""
    for i in enumerate(mat[0]):
        s += str(i[0])
        if i[0] < 10:
            s += " "
    print s
    c = 0
    for row in mat:
        c += 1
        for i in row:
            print i,
        print c
    print ""
    
def clamp_mat(x, y, mat):
    if x < 0:
        x = 0
    elif x >= len(mat):
        x = len(mat) - 1
    if y < 0:
        y = 0
    elif y >= len(mat[0]):
        y = len(mat[0]) - 1
    return x, y

def clone_mat(mat):
    new_mat = []
    for y in mat:
        new_mat.append(list(y))
    return new_mat

def graph(mat, equ, mark="X"):
    for c in range(len(mat)*10):
        if equ(c) in range(len(mat)*10):
            matrx[equ(c/10)][int(c/10)] = mark
    return mat
