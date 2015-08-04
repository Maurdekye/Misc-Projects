import os
import sys
import traceback
from os.path import isdir, join, realpath

def getXML(path, indent=0):
    s = ""
    base = os.path.basename(path)
    printbase = base.replace(" ", "-").replace(".", "_")
    ind = "\t"*indent
    s += ind + "<" + printbase + ">\n"
    if isdir(path):
        for f in os.listdir(path):
            s += getXML(join(path, f), indent+1)
    else:
        with open(path, "r") as f:
            for l in f:
                s += ind + "\t" + l
            if s[-1] != "\n":
                s += "\n"
    s += ind + "</" + printbase + ">\n"
    return s

def walkPath(path, indent=0):
    i = "\t" * indent
    for f in os.listdir(path):
        print i + f
        fpath = join(path, f)
        if isdir(fpath):
            walkPath(fpath, indent+1)

targetpath = os.getcwd()
if len(sys.argv) >= 2:
    targetpath = sys.argv[1]

base = os.path.basename(targetpath)
with open(join(os.path.dirname(targetpath), base + "_compressed.xml"), "w") as f:
    f.write(getXML(targetpath, 0))
