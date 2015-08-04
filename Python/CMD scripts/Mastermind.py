import random

def shuffle(lst):
    for j in xrange(len(lst)):
        for i in xrange(len(lst)):
            rand = random.randrange(len(lst))
            lst[i], lst[rand] = lst[rand], lst[i]
    return lst

def cleanse(phrase, library):
    clist = [l.upper() for l in phrase if l.upper() in library]
    fin = ""
    for item in clist: fin += item
    return fin

# Modifiable Vars

chances = 8
codelength = 4
choices = 6

# Pregame

code = ""

straightlibr = [l for l in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:choices]]
library = shuffle([l for l in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:choices]])
for i in xrange(codelength): code += library[i]

print "The code is {} letters long, and may contain any of the following letters;".format(codelength)
print str(straightlibr)[1:-1]
print "You have a total of {} guesses to figure it out.".format(chances)

# Main Loop

while True:
    # Check if lost
    if chances <= 0:
        raw_input("You have 0 chances left, you lose! The code was '{}'.".format(code))
        break

    # Accept a guess
    rawguess = raw_input("\nMake a guess: ")
    guess = cleanse(rawguess, library)

    # Check for user error
    if guess == "":
        print "I don't understand that."
        continue
    if len(guess) != codelength:
        print "You need to give a guess exactly {} letters long!".format(codelength)
        continue

    # Check if won
    if guess == code:
        raw_input("You guessed the code!")
        break

    # Count how many they got right
    guess = list(guess)
    placed = 0
    for i, (gl, cl) in enumerate(zip(guess, code)):
        if gl == cl:
            placed += 1
            guess[i] = ' '
    contained = 0
    for l in guess:
        if l in code: contained += 1

    chances -= 1

    # Give them information
    toprint = ""
    if contained > 0:
        toprint += "{} {} in the code but incorrectly placed".format(
        contained, "letter was" if contained == 1 else "letters were")
        if placed <= 0:
            toprint += ". "
    if placed > 0:
        if contained > 0:
            toprint += ", and \n{} {} in the correct place. ".format(placed, "other letter was" if placed == 1 else "other letters were")
        else:
            toprint += "{} {} in the correct place. ".format(placed, "letter was" if placed == 1 else "letters were")
    if placed + contained > 0:
        if placed + contained < codelength:
            toprint += "\nThe rest were not in the code."
    else:
        toprint = "None of the letters were in the code."
    print toprint
    print "You have {} {} left.".format(chances, "guess" if chances == 1 else "guesses")
    
    
