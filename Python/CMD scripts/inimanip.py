import re
import os

def parseanyvalue(v):
    try:
        return int(v)
    except ValueError: pass
    try:
        return float(v)
    except ValueError: pass
    tftable = {"true":True, "false":False}
    cleanstr = str(v).lower().strip()
    if cleanstr in tftable:
        return tftable[cleanstr]
    return str(v)

def checkvalid(name):
    if not "." in name or len(name) < 3:
        raise Exception("must be in block-key format (contain a .)")
    return name.split(".", 1)

def getflag(fname, name):
    block, key = checkvalid(name)
    try:
        with open(fname, "r") as f:
            inblock = False
            for line in f.read().split("\n"):
                #print(line)
                if inblock:
                    if line.startswith("["):
                        #print("found block, value nonexistant")
                        return None
                    else:
                        m = re.match(key + "\s*=(.+)", line)
                        #print(True, line, m)
                        if m:
                            return parseanyvalue(m.group(1).strip())
                elif re.search("\[{}\]".format(block), line):
                    #print(False, line, re.search("\[{}\]".format(block), line))
                    inblock = True
            #print("block nonexistant")
            return None
    except IOError:
        #print("file nonexistant")
        return None

def setflag(fname, name, value):
    block, key = checkvalid(name)
    ingestion = ""
    if os.path.isfile(fname):
        with open(fname, "r") as f:
            ingestion = f.read()
    inblock = False
    lines = ingestion.split("\n")
    for lnum in range(len(lines)):
        line = lines[lnum]
        if inblock:
            if line.startswith("["):
                lines.insert(lnum, key + " = " + str(value))
                break
            elif line.startswith(key):
                lines[lnum] = key + " = " + str(value)
                break
        elif line.startswith("[{}]".format(block)):
            inblock = True
    else:
        if not inblock:
            lines.append("")
            lines.append("[{}]".format(block))
        lines.append(key + " = " + str(value))
    while len(lines) > 0 and lines[0] == "":
        lines = lines[1:]
    #print(lines)
    ingestion = "\n".join(lines)
    with open(fname, "w") as f:
        f.write(ingestion)

def getBlock(fname, block):
    pairs = {}
    pairpat = re.compile("(.+)\w*=(.+)")
    try:
        with open(fname, "r") as f:
            inblock = False
            for line in f:
                if inblock:
                    if line.startswith("["):
                        break
                    else:
                        m = pairpat.search(line)
                        if m:
                            pairs[parseanyvalue(m.group(1))] = parseanyvalue(m.group(2))
                elif line.startswith("[{}]".format(block)):
                        inblock = True
    except IOError:
        pass
    return pairs

def allBlocks(fname):
    blocks = []
    blockpat = re.compile("\[(.+)\]")
    try:
        with open(fname, "r") as f:
            for line in f:
                m = blockpat.match(line)
                if m:
                    blocks.append(m.group(1))
    except IOError:
        pass
    return blocks

def allKeys(fname):
    data = {}
    for block in allBlocks(fname):
        data[block] = getBlock(fname, block)
    return data