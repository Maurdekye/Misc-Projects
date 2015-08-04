def groups(nlist):
    fin = [[]]
    for item in nlist:
        if fin[-1] == [] or fin[-1][0] == item:
            fin[-1] += [item]
        else:
            fin += [[item]]
    return fin

def congeal(nlist):
    finstr = ""
    for item in nlist: finstr += str(item)
    return finstr

seed = [1]

for i in range(15):
    print(" "*int(40-len(seed)/2), congeal(seed))
    newls = []
    for nlist in groups(seed):
        newls += [len(nlist), nlist[0]]
    seed = newls
