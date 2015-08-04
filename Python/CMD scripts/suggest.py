def trim(text):
    if text == '': return ''
    def trim_right(txt):
        if txt == '': return ''
        if txt[-1] == ' ':
            return trim_right(txt[:-1])
        return txt
    def trim_left(txt):
        if txt == '': return ''
        if txt[0] == ' ':
            return trim_left(txt[1:])
        return txt
    
    return trim_left(trim_right(text))

def fixword(word):
    if word == '': return word
    if word[-1] == '\n':
        word = word[:-1]
    if word[0] == '\n':
        word = word[1:]
    return trim(word).lower()

lib_import = 'thesaurus.txt'
library = []
with open(lib_import, 'r') as f:
    for line in f:
        library.append(fixword(line))

print '{} words indexed to be searched about.\n'.format(len(library))

deb = 0

maxlen = 0
total = 0
while True:
    usr_in = fixword(raw_input('> '))
    suggestions = {}
    for word in library:
        if deb: print('\ntesting word {}'.format(word))
        for i, letter in enumerate(word):
            if len(usr_in) + i > len(word):
                if deb: print('reached end of word, no match found')
                break
            check = word[i:i+len(usr_in)]
            if deb:
                begin = word[:i]
                end = word[i+len(usr_in):]
                print(
                    "checking section {}'{}'{} of word vs input of '{}'".format(
                        begin, check, end, usr_in))
            if check == usr_in:
                if deb: print('word meets params, adding to suggestions')
                try:
                    suggestions[i]
                except KeyError:
                    suggestions[i] = []
                suggestions[i] += [word]
                maxlen = max(maxlen, i)
                total += 1
                break
        if deb: raw_input()

    for i in xrange(maxlen):
        if i not in suggestions.keys():
            continue
        for word in suggestions[i]:
            print word
            
    print '\n{} Suggestions available for text input.\n'.format(total)
