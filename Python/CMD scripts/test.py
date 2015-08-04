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
    word = word[:cur]
    return word

print trim('         ')

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

for x in xrange(10):
    print prettyprint(xrange(x))
