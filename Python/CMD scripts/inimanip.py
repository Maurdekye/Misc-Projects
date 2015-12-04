import re
import os

def parseanyvalue(v):
    try:
        return int(v)
    except ValueError: pass
    try:
        return float(v)
    except ValueError: pass
    tftable = {"true":True, "false":False, "t":True, "f":False}
    cleanstr = str(v).lower().strip()
    if cleanstr in tftable:
        return tftable[cleanstr]
    return str(v)

def getflag(fname, name):
    if not "." in name:
        raise Exception("must be in block-key format (contains a .)")
    block, key = name.split(".", 1)
    try:
        with open(fname, "r") as f:
            inblock = False
            for line in f.read().split("\n"):
                print(line)
                print("\[{}\]".format(block))
                if inblock:
                    if line.startswith("["):
                        print("found block, value nonexistant")
                        return None
                    else:
                        m = re.match(key + "\w*=(.+)", line)
                        if m:
                            return parseanyvalue(m.group(1))
                elif re.search("\[{}\]".format(block), line):
                    inblock = True
            print("block nonexistant")
            return None
    except IOError:
        print("file nonexistant")
        return None

def setflag(fname, name, value):
    if not "." in name:
        raise Exception("must be in block-key format (contains a .)")
    block, key = name.split(".", 1)
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
    while len(lines) > 0:
        if lines[0] == "":
            lines = lines[1:]
        elif lines[-1] == "":
            lines = lines[-1]
        else:
            break
    print(lines)
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
                        m = pairpat.match(line)
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