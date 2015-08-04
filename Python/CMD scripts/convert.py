import re
import sys
patterns = [
    ("(?![a-zA-Z])range\((.*)\)", "list(range(\1))"),
    ("xrange", "range"),
    ("print (.*),\s*^", "print(\1, end=\"\")"),
    ("print (.*)", "print(\1)"),
    ("raw_input", "input")
    ]

patterns_compiled = [(re.compile(p), r) for p, r in patterns]
if len(sys.argv) >= 1:
    buffer = ""
    try:
        with open(sys.argv[0], "r") as f:
            buffer = f.read()
        for pat, rep in patterns_compiled:
            buffer = re.sub(pat, rep, buffer)
        print(buffer)
        ans = input("\n\nIs this good? (y/n)")
        if ans is "y":
            with open(sys.argv[0], "w") as f:
                f.write(buffer)
            input("success; file overwritten")
        else:
            input("failure; user abort")
    except:
        input("failure; filename invalid")
else:
    input("failure; no filename given")