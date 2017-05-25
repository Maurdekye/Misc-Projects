import re
import os

words = []

with open("/usr/share/dict/words", "r") as f:
    for w in f:
        words.append(w)

vowelpattern = re.compile("([aeiou]+)")

sort = sorted(words, key=lambda w: max([len(g) for g in vowelpattern.search(w).groups()[1:] if len(vowelpattern.search(w).groups()) > 1]))
