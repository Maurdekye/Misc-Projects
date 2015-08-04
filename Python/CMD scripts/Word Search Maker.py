import random
import sys
import os

def compress(ls):
    ret = ''
    for l in ls: ret += l
    return ret

def addtab(word, limit):
    return word + ' ' * max(0, limit - len(word))

def shuffle(ls):
    for i in range(100):
        for e in range(len(ls)):
            swap = random.randrange(len(ls))
            ls[e], ls[swap] = ls[swap], ls[e]
    return ls


letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
           'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
           'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
           ]

# Defines the size and name of the word search

name = input('Name of word search: ')
size = int(input('Size of word search: '))
while size < 4:
    size = int(input('Word grid too small, must be at least 4: '))

# Creates the initial grid

cross = [[' ' for i in range(size)] for j in range(size)]
words = []

wordfile = None

# Testing to see if there is a 'words.txt' file available to take words froms

if len(sys.argv) <= 1:

    # Takes words given by the user

    msg = '''
Please add words below.
Must be at least 4 letters and fit
in the size of the word search.
Enter a blank word to finish.

'''

    while True:
        wrd = input(msg)
        if len(wrd) == 0:
            if len(words) == 0:
                input('Cancelling word search.\n\nPress enter to exit the program.')
                sys.exit()
            else:
                print('Finishing word search...')
                break
        if len(wrd) < 4:
            print('Word too short.')
            continue
        if len(wrd) > size:
            print('Word too long.')
            continue
        new = compress([l.upper() for l in wrd if l.upper() in letters])
        print("Added word '{}'.\n".format(new))
        words += [new]
else:

    # Uses words from given file if available

    print('Using words from given file...')

    try:
        with open(sys.argv[1]) as wordfile:
            words = [compress([l.upper() for l in word if l.upper() in letters]) for word in wordfile.read().split('\n')]
    except:
        input("File given was invalid.")
        sys.exit()

orients = [
    (1, 1),
    (1, 0),
    (0, 1),
    (1, -1)
    ]

# Places words on the grid

deb = False
if 'DEBUGME' in words:
    deb = True
    print('Word Search debug mode on.')

for word in words:
    
    if len(word) < 4 or len(word) > size:
        print("Word '{}' is a bad size, leaving it out".format(word))
        words.remove(word)
        continue

    leave = True
    lim = max(min(size - len(word) + 1, size - 1), 0)
    if deb: print(lim)
    for ori in orients:
        for x, y in shuffle([(b, c) for b in range(lim) for c in range(lim)]):
            leave = True
            px, py = x, y
            stay = False
            for l in word:
                if deb: print(py, px, ':', len(cross), len(cross[0]))
                if cross[py][px] not in [l, ' ']:
                    stay = True
                    leave = False
                    break
                if cross[py][px] ==  ' ': continue
                px += ori[0]
                py += ori[1]
            if stay: continue
            for l in word:
                cross[y][x] = l
                x += ori[0]
                y += ori[1]
            if leave: break
        if leave: break
    else:
        for ori in orients:
            for x, y in shuffle([(b, c) for b in range(lim) for c in range(lim)]):
                leave = True
                px, py = x, y
                stay = False
                for l in word:
                    if deb: print(py, px, ':', len(cross), len(cross[0]))
                    if cross[py][px] not in [l, ' ']:
                        stay = True
                        leave = False
                        break
                    px += ori[0]
                    py += ori[1]
                if stay: continue
                for l in word:
                    cross[y][x] = l
                    x += ori[0]
                    y += ori[1]
                if leave: break
            if leave: break
        else:
            print("Couldn't find room for '{}', leaving it out.".format(word))
            words.remove(word)

# Fills in remaining grid spaces with random letters

for x in range(len(cross)):
    for y in range(len(cross[x])):
        if cross[x][y] == ' ':
            if not deb:
                cross[x][y] = random.choice(letters)

print('Writing down word search...')

# Opens the file to write the word search into

if not os.path.exists('Word Searches'): os.makedirs('Word Searches')

with open('Word Searches/' + name + '.txt', 'w') as f:
    def write(txt):
        f.write(txt + '\n')
    center = size/2 + len(name)/2
    write(' '*int(center) + name) # Writes the title
    write('\n')
    for line in cross:
        l = ''
        for let in line:
            l += let + ' '
        write(l) # Writes each line of the grid
    write('\n')
    tabspace = max([len(w) for w in words]) + 4
    line = ''
    for word in words:
        line += addtab(word, tabspace)
        if len(line) > size + tabspace:
            write(line) # Writes the words to search for
            line = ''
    if line != '': write(line)

input("Word search written to 'Word Searches/{}.txt'. \n\nPress enter to end the program.".format(name))
