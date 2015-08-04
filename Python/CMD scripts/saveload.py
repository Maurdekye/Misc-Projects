def savelist(filename, ls):
    with open(filename+".txt", "w") as f:
        for key in ls:
            f.write(str(key)+":"+str(ls[key])+"\n")

def loadlist(filename):
    with open(filename+".txt", "r") as f:
        ls = {}
        for line in f:
            items = line.split(":")
            try: items[1] = int(items[1])
            except ValueError:
                if items[1][-1:] == "\n":
                    items[1] = items[1][:-1]
            ls[items[0]] = items[1]
    return ls
