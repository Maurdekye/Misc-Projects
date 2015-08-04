import sys

f = None

try:
    f = open('readthis.txt', 'r')
except IOError:
    print "Need file named 'readthis.txt' to read from."
    sys.exit()

content = f.read()

if content == '':
    print "File 'readthis.txt' is empty."
    sys.exit()

def prettyprint(ls):
    if len(ls) == 0:
        return 'Nothing.'
    if len(ls) == 1:
        return str(ls[0]) + '.'
    if len(ls) == 2:
        return str(ls[0]) + ' and ' + str(ls[1]) + '.'
    fin = ''
    for w in list(ls)[:-1]:
        fin += str(w) + ', '
    return fin + 'and ' + str(ls[-1]) + '.'

def trim(word):
    if word == '': return ''
    cur = 0
    while word[cur] == ' ':
        cur += 1
        if cur == len(word):
            return ''
    word = word[cur:]
    cur = len(word) - 1
    while word[cur] == ' ':
        cur -= 1
    cur += 1
    word = word[:cur]
    return word

def valid(word):
    for l in word:
        if l not in [' ', '\n']: return True
    return False

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
           'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
           'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

letters += [l.lower() for l in letters] + ["'", '-']

split = ['']
cur = 0
onword = False
for l in content:
    if l not in letters and onword:
        cur += 1
        split += ['']
        onword = False
    else:
        onword = True
        split[cur] += l

split = [trim(w) for w in split if valid(w)]

print "{} letters in the file, making {} words.".format(len(content), len(split))

lengths = [len(w) for w in split]
avg = sum(lengths) / float(len(split))
rng = max(lengths) - min(lengths)
print "Average word length is {}, with a range of {}.".format(avg, rng)

longest, shortest = split[0], split[0]
for word in split[1:]:
    if len(word) > len(longest): longest = word
    if len(word) < len(shortest): shortest = word
print "The longest word is '{}', and the shortest '{}'.".format(longest, shortest)
print
print content
