import sys
from random import choice

Dictionary = 'thesaurus.txt' # Dictionary of words the program will guess from
debug = False # Enables extra Debug information
listpossible = False # Displays all possible words between each letter guess

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

def recur_getnum(quest='Please give a number:', wrong="I didn't understand that."):
    usr_in = input(quest + ' ')
    try:
        to_sender = int(usr_in)
    except ValueError:
        print(wrong)
        return recur_getnum(quest, wrong)
    return to_sender
    
def recur_yorn(quest='Yes or No?', wrong="I didn't understand that."):
    valid_yes = ['yes', 'yep', 'yeah', 'yiss', 'yeh', '*nod head*', 'y']
    valid_no = ['no', 'nope', 'nuh uh','nuh-uh', 'nah', '*shake head*', 'n']
    usr_in = input(quest + ' ').lower()
    for yes, no in zip(valid_yes, valid_no):
        if yes in usr_in:
            return True
        if no in usr_in:
            return False
    print(wrong)
    return recur_yorn(quest, wrong)

# --- Initial Variable Declarations and Preparations --- #

words = []
with open(Dictionary, 'r') as f:
    for line in f:
        if line[-1] == '\n':
            line = line[:-1]
        words += [trim(line.lower())]

word_len = recur_getnum('How long is this word?', 'Please give a number.')

new_words = []
for word in words:
    if len(word) == word_len:
        new_words += [word]

alpha = [
    'a', 'b', 'c', 'd', 'e', 'f',
    'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r',
    's', 't', 'u', 'v', 'w', 'x',
    'y', 'z']

miss_phrases = [
    "Dang.",
    "Aw dang.",
    "Oh well.",
    "Too bad.",
    "I guessed wrong.",
    "Maybe next letter.",
    ":'(",
    "I can do better.",
    "What's this? An incorrect letter??",
    "How could this happen?",
    "What are you thinking of?",
    "I WILL guess this word.",
    "Just one more chance?",
    "I need a moment to think.",
    "Can I have a hint?",
    "I wish I knew.",
    "I have chosen... poorly.",
    ".....Interesting.",
    "Hm, that gives me an idea.",
    "There goes that theory.",
    "I should know this."
    ]

used = []
missed = []
chances = 6

guess = ['_' for x in range(word_len)]

# --- Main Loop --- #

while True:
    if debug: print('begin')

    # --- Reformat List by Letters Known --- #

    old_words = list(new_words)
    new_words = []
    for word in old_words:
        for letter, check in zip(word, guess):
            if check == '_':
                if letter in used:
                    break
                continue
            if letter != check:
                break
        else:
            new_words += [word]

    if debug:
        for word in new_words:
            print(word)
        print('len words ' + str(len(new_words)))

    if listpossible:
        print('List of Possible Words:')
        for w in new_words:
            print(w)
        print('\n' + str(len(new_words)) + ' in total.')
    
    # --- Check List for Answer --- #
    
    if len(new_words) == 1:
        ans = recur_yorn("Is your word '{}'?".format(new_words[0]))
        if ans:
            input('Yes, I win!')
        else:
            input('I do not know of any word by the parameters you have provided.')
        sys.exit()

    if len(new_words) == 0:
        print('I do not know of any word by the parameters you have provided.')
        input()
        sys.exit()

    # --- Letter Frequency Indexing --- #
    
    priorities = [[] for x in range(word_len)]
    for letter in range(word_len):
        if debug: print('\n\nletter ' + str(letter))
        if debug: print('\t\t' + guess[letter])
        if guess[letter] != '_':
            continue
        counts = {l : 0 for l in alpha if l not in used}
        for word in new_words:
            if word[letter] not in used and word[letter] in alpha:
                counts[word[letter]] += 1
        counts = {l : c for l, c in counts.items() if c > 0}
        
        if debug:
            print('\n' + str(counts))

        fin_dict = []
        for comp_letter in counts:
            for i, (number, ltr) in enumerate(fin_dict):
                if counts[comp_letter] < number:
                    fin_dict.insert(i, (counts[comp_letter], comp_letter))
                    break
            else:
                fin_dict.append((counts[comp_letter], comp_letter))
                
        priorities[letter] = [(a, b) for a, b in fin_dict[::-1]]

    if debug:
        for lis in priorities:
            print(lis)

    # --- Letter Guess --- #

    for letter in guess: print(letter.upper(), end=' ')
    print('\n')

    frees = [i for i in range(word_len) if guess[i] == '_']
    guessletter = (0, 'a')
    for i in frees:
        if priorities[i][0][0] > guessletter[0]:
            guessletter = priorities[i][0]
    guessletter = guessletter[1]
    if not recur_yorn('Is the letter {} in the word?'.format(guessletter.upper())):
        missed += [guessletter]
        print('\n' + choice(miss_phrases))
        chances -= 1
        if chances <= 0:
            input("I've run out of chances!")
            sys.exit()
        else:
            print('I have {} chances left.'.format(chances))
    else:
        
        while True:
            found = []
            fin, numbers = '', ''
            for i, letter in enumerate(guess):
                if letter == '_':
                    fin += '_ '
                    numbers += str(i+1) + ' '
                    if i >= 9:
                        fin += ' '
                else:
                    fin += letter.upper() + ' '
                    numbers += '  '
                    found += [i]
                    
            print(fin)
            print(numbers)
            print() 
            usr_plin = recur_getnum("What index is it in?") - 1
            if usr_plin > word_len or usr_plin < 0:
                print("That doesn't make sense.")
                continue
            if usr_plin in found:
                print("There's already a letter there.")
                continue
            guess[usr_plin] = guessletter
            if recur_yorn("Does it appear anywhere else?"):
                continue
            break
    used += [guessletter]
    print()
    
    
    
