import random

# Pre-Variable Functions -----------------------------------------------------------

def blank_mat(size=10, typemat="-"): return [[typemat for i in range(size)] for i in range(size)]

def print_mat(mat):
    c = 0
    for i in range(len(mat)):
        print(i+1, end="")
    print("")
    for i in mat:
        for j in i:
            print(j, end="")
        c+=1
        print(c)
    
def clamp_mat(x, y, mat):
    if x < 0:
        x = 0
    elif x >= len(mat):
        x = len(mat) - 1
    if y < 0:
        y = 0
    elif y >= len(mat[0]):
        y = len(mat[0]) - 1
    return x, y

def change_mat(x, y, changeTo, mat):
    x, y = clamp_mat(x, y, mat)
    mat[y][x] = changeTo[0]
    return mat

def change_player(cply):
    if cply == 1:
        return 0
    return 1

def whitespace(lines=300):
    for i in range(lines):
        print("")

def clamp(minim, num, maxim):
    if num < minim:
        return minim
    if num > maxim:
        return maxim
    return num

def clone(mat):
    new_mat = []
    for y in mat:
        new_mat.append(list(y))
    return new_mat

def makeway(cply):
    whitespace()
    input("Please leave for player "+["two","one"][cply]+".")
			
def recurnum(message):
    num = input(message)
    try:
        num = int(num)
    except ValueError:
        print("Sorry, I didn't understand that.")
        return recurnum(message)
    return num

# Variable Declarations ------------------------------------------------------------------

winner = False
cur_player = 0
player_mats = [[blank_mat(), blank_mat()],[blank_mat(), blank_mat()]]
player_ships = [{},{}]
player_shiphits = [{},{}]

hit_phrases = [
    "You hit something!",
    "You hit an enemy ship!",
    "Your missile landed on an enemy ship!",
    "You guessed correctly!"]

miss_phrases = [
    "You missed.",
    "You missed the enemy's ships.",
    "Your missile landed in the water.",
    "You guessed wrong."]

ship_temp = {
    "aircraft carrier" : 5,
    "battleship" : 4,
    "destroyer" : 3,
    "submarine" : 3,
    "patrol boat" : 2}

# Post-Variable Functions --------------------------------------------------------------

def print_grids(player):
    print_mat(player_mats[player][0])
    print_mat(player_mats[player][1])
    print("")

def place_ships(player):
    for ship in ship_temp:
        ans = 'n'
        player_ships[player][ship] = []
        player_shiphits[player][ship] = []
        while ans == 'n':
            whitespace()
            print_mat(player_mats[player][1])
            print("\nPlayer %d, please place your %s. (%d long)\n" % (player+1, ship , ship_temp[ship]))
            x = clamp(0, recurnum("Choose the X position: ") - 1, len(player_mats[player][1])-1)
            y = clamp(0, recurnum("Choose the Y position: ") - 1, len(player_mats[player][1])-1)
            align = input("Vertical or Horizontal? (v/h): ").lower()
            while not align in ['v','h']:
                print("Please say again, I didn't understand that.")
                align = input("Vertical or Horizontal? (v/h): ").lower()
            if align == 'h':
                x = clamp(0,x,len(player_mats[player][1])-ship_temp[ship])
            elif align == 'v':
                y = clamp(0,y,len(player_mats[player][1])-ship_temp[ship])
            isokay = False
            for i in range(ship_temp[ship]):
                if align == 'h':
                    if player_mats[player][1][y][x+i] in [sh[0].upper() for sh in ship_temp]:
                        break
                if align == 'v':
                    if player_mats[player][1][y+i][x] in [sh[0].upper() for sh in ship_temp]:
                        break
            else:
                isokay = True
            if not isokay:
                input("\nThere is already another ship there. Please choose again.")
                continue
            whitespace()
            old_mat = clone(player_mats[player][1])
            for i in range(ship_temp[ship]):
                if align == 'h':
                    player_mats[player][1][y][x+i] = "#"
                if align == 'v':
                    player_mats[player][1][y+i][x] = "#"
            print_mat(player_mats[player][1])
            ans = input("\nThis is where your "+ship+" will be. Is this good? (y/n): ").lower()
            while not ans in ['y','n']:
                print("Please say again, I didn't understand that.")
                ans = input("This is where your "+ship+" will be. Is this good? (y/n): ").lower()
            if ans == 'n':
                player_mats[player][1] = clone(old_mat)
            else:
                for y in range(len(player_mats[player][1])):
                    for x in range(len(player_mats[player][1][y])):
                        if player_mats[player][1][y][x] == "#":
                            player_mats[player][1][y][x] = ship[0].upper()
                            player_ships[player][ship].append([x,y])

# Main Program --------------------------------------------------------------------------

place_ships(0)
makeway(0)
place_ships(1)
makeway(1)

while not winner:
    whitespace()
    grid = player_mats[cur_player][1]
    e_grid = player_mats[change_player(cur_player)][1]
    view_e_grid = player_mats[cur_player][0]
    print_grids(cur_player)
    print("Player %s's turn." % (["one","two"][cur_player]))
    g_x = recurnum("Guess an X position: ")
    g_y = recurnum("Guess a y position: ")
    whitespace()
    g_x, g_y = clamp_mat(g_x - 1, g_y - 1, e_grid)
    guessed_there = False
    if view_e_grid[g_y][g_x] != "-":
        guessed_there = True
        
        print_grids(cur_player)
    elif e_grid[g_y][g_x] in [sh[0].upper() for sh in ship_temp]:
        prl = [False, False, False]
        player_mats[cur_player][0][g_y][g_x] = "X"
        shipToDel = ""
        for ship in player_ships[change_player(cur_player)]:
            for cord in player_ships[change_player(cur_player)][ship]:
                if cord == [g_x,g_y]:
                    player_mats[change_player(cur_player)][1][g_y][g_x] = ship[0].lower()
                    player_shiphits[cur_player][ship].append(cord)
                    player_ships[change_player(cur_player)][ship].remove(cord)
                    if player_ships[change_player(cur_player)][ship] == []:
                        prl[0] = True
                        shipToDel = ship
                        for cval in player_shiphits[cur_player][ship]:
                            player_mats[cur_player][0][cval[1]][cval[0]] = ship[0].lower()
                            player_mats[change_player(cur_player)][1][cval[1]][cval[0]] = ship[0].lower()
                        del player_shiphits[cur_player][ship]
        if shipToDel != "":
            del player_ships[change_player(cur_player)][shipToDel]
            remaining = len(player_ships[change_player(cur_player)])
            prl[1] = True
            if remaining <= 0:
                prl[2] = True
                winner = True
        print_grids(cur_player)
        if not prl[0]: print(hit_phrases[random.randrange(len(hit_phrases))])
        if prl[0]: print("You sunk player %s %s!" % (["two's","one's"][cur_player], shipToDel))
        if prl[1]: print("player %s now has %d ships left." % (["two","one"][cur_player],remaining))
        if prl[2]: print("\nPlayer %s has won!\n" % (["one","two"][cur_player]))
        if guessed_there: print("You already guessed there, please go again.")
        input("")
    else:
        player_mats[change_player(cur_player)][1][g_y][g_x] = "+"
        player_mats[cur_player][0][g_y][g_x] = "+"
        print_grids(cur_player)
        input(miss_phrases[random.randrange(len(miss_phrases))])
    if not winner:
        makeway(cur_player)
    if not guessed_there:
        cur_player = change_player(cur_player)
