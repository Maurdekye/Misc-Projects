import os

def alterfile(fname, action=lambda line:line):
    bpref = "TEMP"
    tind = 0
    tfname = fname + "~" + bpref
    while os.path.isfile(tfname):
        tfname = fname + "~" + bpref + tind
        tind += 1
    with open(tfname, "a") as tf:
        with open(fname, "r") as f:
            for line in f:
                nline = action(line.rstrip())
                if nline != None:
                    tf.write(str(nline) + "\n")
    os.remove(fname)
    os.rename(tfname, fname)

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

class Ini:
    def __init__(my, fname):
        my.changes = {}
        my.cache = {}
        my.removals = {}
        my.fname = fname
        if os.path.isfile(fname):
            try:
                open(fname, "r").close()
            except IOError:
                raise Exception("File is unreadable.")
        else:
            try:
                open(fname, "a").close()
            except IOError:
                raise Exception("File cannot be created")

    def _retrieve(my, block, key):
        inblock = False
        keyfetchpat = re.compile(key + "\s*=(.+)")
        blockfetchpat = re.compile("\[{}\]".format(block))
        with open(my.fname, "r") as f:
            for line in f:
                if inblock:
                    if line.startswith("["):
                        return None
                    match = keyfetchpat.match(line)
                    if match:
                        return match.group(1)
                elif blockfetchpat.match(line):
                    inblock = True
        return None

    def set(my, location, newvalue):
        block, key = checkvalid(location)
        my.changes[block][key] = newvalue
        my.cache[block][key] = newvalue

    def get(my, location):
        block, key = checkvalid(location)
        if block in cache and key in cache[block]:
            return cache[block][key]
        else:
            value = my._retrieve(block, key)
            if value != None:
                cache[block][key] = value
            return value

    def remove(my, location):
        block, key = checkvalid(location)

