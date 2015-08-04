import sys
from random import randrange
import copy
try:
    import pygame as pyg
except ImportError:
    input("You need to have Pygame for Python 2.7.X installed to run this program!\n")
    sys.exit()

"""
 --- Formatting Rules ---
     * Store questions in a text file called "Jeopardy Questions.txt".
         * Type category headers as lines without indents.
         * Type questions as lines that begin with with four spaces or a tab.
             * Questions are added to the most recently typed category.
             * Information is separated by a colon. All questions must have at least
                 one colon. The hint comes before the first one, and the question
                 contains everything after.
                 * Questions without a single colon will not be registered.
                 * All colons after the first one will be accepted as text part of
                     the question
                 * To create colons in the Hint, use '&c'.
                     * the Question will also translate '&c' to a colon, but it is not
                         neccessary to use the symbol.
         * You can have categories without any questions in them (for some reason),
             but any questions typed without a parent category won't be registered.
         * Point values will be in the order that they are added, the first one under each
             category being worth 100 and increasing in increments of 100 for each
             following question.
             * If you want, you can add a custom point value to individual questions
                 by including it within a set of double angle-brackets anywhere in
                 the line, but it must be in front of the tab / four space indent.
                 * If you add multiple sets of double angle-brackets into the
                     line, it will use the last one present on the line as the
                     score for the question.
                 * Do note that the program already adds a daily double point
                     value question to the board which is completely random,
                     anonymous, and with the neccessary extra visual accompanyment.
         * Example:
Music
    My favorite music. : What is music written by the Beatles?
    The best song not by the beatles. : What is Stairway <<50>> to Heaven? 
Dance Moves!!
Movies
    <<200>> Main character in the movie 'The Hunger Games&c Catching Fire'. : Who is Katniss Everdeen?
    Movie where a short person tries to throw a peice of jewlery into an active volcano. : What is The Hobbit: An Unexpected Journey?
Videogames
    The only game with both assassination and pirates! : What is Assassin's Creed IV&c Black Flag? <<99999>>

     * Store team names in a text file called "Jeopardy Teams.txt".
         * Team names are separated by new lines
     * Do not put too many categories or teams, it may get cramped and render improperly.
"""

class Question:
    taken = False
    double = False
    def __init__(this, question, answer, value=None):
        this.question = question
        this.answer = answer
        this.value = value

class Category:
    def __init__(this, name, questions):
        this.name = name
        this.questions = questions
        for i in range(len(this.questions)):
            if this.questions[i].value == None:
                this.questions[i].value = (i+1)*100

def drawtext(screen, text, pos, size=25, color=(0, 0, 0)):
    screen.blit(pyg.font.SysFont("monospace", int(size)).render(str(text), 0, color), pos)

def justify(text, limit=20):
    fin = ""
    wid = 0
    for w in text.split(" "):
        if len(w) + wid >= limit:
            fin += '\n'
            wid = len(w)
        else:
            fin += ' '
            wid += len(w) + 1
        fin += w
    if fin[-1] == '\n':
        fin = fin[:-1]
    return fin[1:]
    
pyg.init()

WIDTH, HEIGHT = pyg.display.list_modes()[0]
scrn = pyg.display.set_mode((WIDTH, HEIGHT), pyg.FULLSCREEN)
clicked = {i : False for i in range(10)}

textsize = int(min(WIDTH, HEIGHT) / 24)
backgroundcolor = (0, 160, 190)
highlightcolor = (100, 220, 250)
takencolor = (200, 80, 120)
bordercolor = (0, 80, 95)
textcolor = (0, 60, 40)
teams = {}
curturn = 0

try:
    with open("Jeopardy Teams.txt", "r") as f:
        teams = {}
        for l in f:
            teams[l] = 0
except IOError:
    teams = {t : 0 for t in [
        "Team 1",
        "Team 2",
        "Team 3",
        "Team 4"
    ]}

dev = False
board = []
try:
    with open("Jeopardy Questions.txt", "r") as f:
        categ = ""
        queses = []
        for line in f:
            if line[:4] == "    " or line[0] == "\t":
                v = None
                assess = ""
                begin = 0
                for i in range(len(line)-1):
                    f = line[i:i+2]
                    if f == "<<":
                        begin = i
                        i += 2
                        while i < len(line):
                            if line[i:i+2] == ">>":
                                line = line[:begin] + line[i+3:]
                                try:
                                    v = int(assess)
                                except ValueError: pass
                                assess = ""
                                break
                            else:
                                assess += line[i]
                            i += 1
                inp = line[1:].split(":")
                if line[:4] == "    ": inp = line[4:].split(":")
                if len(inp) <= 1: continue
                h = inp[0].replace("&c", ":")
                if h[-1] == " ": h = h[:-1]
                q = ":".join(inp[1:]).replace("&c", ":")
                if q[0] == " ": q = q[1:]
                queses += [Question(h, q, v)]
            else:
                if categ != "":
                    board += [Category(categ, copy.deepcopy(queses))]
                queses = []
                categ = line
        board += [Category(categ, copy.deepcopy(queses))]
except IOError:
    dev = True
    board = [
        Category("This is a default Jeopardy board", [
            Question("Person this game was created by.", "Who is Nicholas Colaprete?", 10000)
        ]),
        Category("Because the text file, 'Jeopardy Questions.txt' could not be parsed / found.", []),
        Category("Animals", [
            Question("Fast and Spotted.", "What is a cheetah?"),
            Question("Fat and pink.", "What is a pig?"),
            Question("Sleek and domesticated.", "What is a cat?"),
            Question("Green and slimy.", "What is a frog?"),
            Question("Big and grey.", "What is an elephant?")
        ]),
        Category("Computer Cables", [
            Question("Used for the internet.", "What is an ethernet cable?"),
            Question("Used to charge your phone.", "What is a micro universal serial bus cable?"),
            Question("Connects Perihperals.", "What is a universal serial bus cable?"),
            Question("Connects to your monitor.", "What is a digital video or HDMI cable?"),
            Question("Connects to your headphones.", "what is an audio jack?")
        ])
    ]
    
if dev:
    board[0].questions[0].double = True
else:
    sel = board[randrange(len(board))]
    for i in range(1000):
        if len(sel.questions) > 0: break
        sel = board[randrange(len(board))]
    sel.questions[randrange(len(sel.questions))].double = True

hmul = (WIDTH/len(board))
topbuff = max([len(justify(c.name, hmul/(textsize*.75)).split('\n')) for c in board])*(textsize + 5)
tsplit = [(t, justify(t, (float(WIDTH)/len(teams)) / textsize/.6 - 2).split('\n')) for t in teams]
splits = [t[1] for t in tsplit]
lowbuff = max([(len(c)+1)*textsize for c in splits])
cols = max([len(c.questions) for c in board])

vmul = 0
try:vmul = (HEIGHT-topbuff-lowbuff)/cols
except: pass
highlighted = [0, 0]
hcat = Category("null", [])
hques = Question("null", "null")
active = None
scene = 0
winning = 0
tempo = 0

while True:
    mpos = pyg.mouse.get_pos()
    mx, my = mpos
    hx, hy = highlighted
    tempo += 1
    
    # --- Events
    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()
        if ev.type == 2:
            if ev.key == 27:
                active = None
                scene = 0
        if ev.type == 5:
            clicked[ev.button] = True

    # --- Renders
    if hques.double and active != None:
        scrn.fill((220, 220, 220))
    else:
        scrn.fill(backgroundcolor)
    pyg.draw.rect(scrn, bordercolor, [0, 0, 2, HEIGHT])
    pyg.draw.rect(scrn, bordercolor, [WIDTH-2, 0, 2, HEIGHT])
    pyg.draw.rect(scrn, bordercolor, [0, 0, WIDTH, 2])
    pyg.draw.rect(scrn, bordercolor, [0, HEIGHT-2, WIDTH, 2])

    # -- Main Screen
    if active == None and scene == 0:

        # - Check if all are answered
        for c in board:
            for q in c.questions:
                if not q.taken:
                    break
            else:
                continue
            break
        else:
            scene = 2
            continue
            
        # - Render Highlighted Question
        if mx >= hx*hmul and mx <= (hx+1)*hmul and my >= hy*vmul+topbuff and my <= (hy+1)*vmul+topbuff:
            if not hques.taken:
                pyg.draw.rect(scrn, highlightcolor, [hx*hmul, hy*vmul+topbuff, hmul, vmul])
                if clicked[1]:
                    active = hques
        else:
            for x in range(len(board)):
                for y in range(cols):
                    if mx < x*hmul: continue
                    if mx > (x+1)*hmul: continue
                    if my < y*vmul+topbuff: continue
                    if my > (y+1)*vmul+topbuff: continue
                    if y >= len(board[x].questions): continue
                    hcat = board[x]
                    hques = hcat.questions[y]
                    highlighted = [x, y]
                    if not hques.taken:
                        pyg.draw.rect(scrn, highlightcolor, [x*hmul, y*vmul+topbuff, hmul, vmul])
                    break
                else:
                    continue
                break

        # - Render Categories
        for i, cat in enumerate(board):
            pyg.draw.rect(scrn, bordercolor, [i*hmul, topbuff, hmul, 2])
            for j, ques in enumerate(cat.questions):
                if ques.taken:
                    hcol = takencolor
                    if ques.double: hcol = (220, 220, 220)
                    pyg.draw.rect(scrn, hcol, [i*hmul, j*vmul+topbuff, hmul, vmul])
                pyg.draw.rect(scrn, bordercolor, [i*hmul, (j*vmul) + topbuff, hmul, 2])
                pyg.draw.rect(scrn, bordercolor, [i*hmul, (j*vmul) + topbuff+vmul, hmul, 2])
                drawtext(scrn, ques.value,
                         [i*hmul + (hmul/2 - textsize*len(str(ques.value))/2),
                          j*vmul + topbuff + (vmul/2 - textsize/4)], textsize, textcolor)
            for j, t in enumerate(justify(cat.name, hmul/(textsize*0.75)).split("\n")):
                drawtext(scrn, t, [i*hmul + hmul/2 - (len(t)*(textsize/2))/2,
                                   5+(j*(textsize+5))], textsize, textcolor)
            pyg.draw.rect(scrn, bordercolor, [i*hmul, 0, 2, len(cat.questions)*vmul + topbuff])
            pyg.draw.rect(scrn, bordercolor, [(i+1)*hmul, 0, 2, len(cat.questions)*vmul + topbuff])

        # - Render Teams / Scores
        for i, (team, ts) in enumerate(tsplit):
            spacing = WIDTH / len(teams)
            leftbuff = spacing*i
            if i == curturn:
                pyg.draw.rect(scrn, (0, 255, 0), [leftbuff + (spacing/4) - 5, HEIGHT-textsize - 5, spacing/2 + 10, 12])
            for j, t in enumerate(ts):
                drawtext(scrn, t,
                         [leftbuff + (spacing/2) - (len(t)*textsize*.6)/2,
                          HEIGHT-(len(ts)-j+1)*textsize],
                         textsize, textcolor)
                pyg.draw.rect(scrn, bordercolor, [leftbuff + (spacing/4), HEIGHT-textsize, spacing/2, 2])
                drawtext(scrn, teams[team],
                         [leftbuff + (spacing/2) - (len(str(teams[team]))*textsize*.6)/2,
                          HEIGHT-textsize], textsize, textcolor)
                            

    # -- Individual Question
    elif scene in [0, 1]:
        
        # - Render Team Selection
        hteam = ""
        if scene == 1:
            for i in range(len(teams)):
                left = (float(i)/len(teams))*WIDTH
                right = left + (WIDTH/len(teams))
                up = HEIGHT-lowbuff
                a = mx > left
                b = mx < right
                c = my > up
                d = my < HEIGHT
                if a and b and c and d:
                    pyg.draw.rect(scrn, highlightcolor, [left+2, up+2,
                                                         WIDTH/len(teams)-4, lowbuff-4])
                    hteam = tsplit[i][0]
                    break
            
        # - Render Teams / Scores
        for i, (team, ts) in enumerate(tsplit):
            spacing = WIDTH / len(teams)
            leftbuff = spacing*i
            for j, t in enumerate(ts):
                drawtext(scrn, t,
                         [leftbuff + (spacing/2) - (len(t)*textsize*.6)/2,
                          HEIGHT-(len(ts)-j+1)*textsize],
                         textsize, textcolor)
                pyg.draw.rect(scrn, bordercolor, [leftbuff + (spacing/4), HEIGHT-textsize, spacing/2, 2])
                drawtext(scrn, teams[team],
                         [leftbuff + (spacing/2) - (len(str(teams[team]))*textsize*.6)/2,
                          HEIGHT-textsize], textsize, textcolor)
        
        # - Render Title
        qtitle = justify(hcat.name, WIDTH/textsize/2).split("\n")
        for j, t in enumerate(qtitle):
            drawtext(scrn, t,
                     [(WIDTH/2 - len(t)*textsize*0.9) / 2, 5+(j*(1.5*textsize+5))],
                     textsize*1.5, textcolor)
        drawtext(scrn, "-",
                 [WIDTH/2, (len(qtitle)/2.0)*textsize*1.5 - (textsize/2)],
                 textsize*1.5, textcolor)
        drawtext(scrn, str(hques.value) + {1:"x2", 0:""}[hques.double],
                 [WIDTH*.75 - len(str(hques.value))*textsize*.9,
                  len(qtitle)/2.0 * (textsize*1.5) - textsize/2],
                 textsize*1.5, textcolor)

        # - Render Question / Answer
        utext = hques.question
        if scene == 1:
            utext = hques.answer
            if not hques.taken:
                hques.taken = True
                curturn = (curturn+1)%len(teams)
        qques = justify(utext, WIDTH/textsize).split("\n")
        for j, t in enumerate(qques):
            h = len(t)/2
            drawtext(scrn, t,
                     [WIDTH/2 - h*textsize*.9, HEIGHT/2 + j*textsize*1.5],
                     textsize*1.5, textcolor)
        if hques.double:
            drawtext(scrn, "DOUBLE",
                     [WIDTH/10, len(qtitle)*textsize*1.5 + textsize],
                     textsize*2,(110, 110, 110))
            drawtext(scrn, "DOUBLE",
                     [WIDTH*6/10, len(qtitle)*textsize*1.5 + textsize],
                     textsize*2,(110, 110, 110))

        if clicked[1]:
            if scene == 0:
                scene = 1
            else:
                if hteam != "":
                    teams[hteam] += hques.value
                    if hques.double: teams[hteam] += hques.value
                    hques.taken = True
                    winning = max(teams.values())
                    active = None
                    scene = 0

    # -- Final Screen
    else:
        # - Render Teams / Scores
        for i, (team, ts) in enumerate(tsplit):
            spacing = WIDTH / len(teams)
            leftbuff = spacing*i
            for j, t in enumerate(ts):
                drawtext(scrn, t,
                         [leftbuff + (spacing/2) - (len(t)*textsize*.6)/2,
                          HEIGHT/2-(len(ts)-j+1)*textsize],
                         textsize, textcolor)
            pyg.draw.rect(scrn, bordercolor, [leftbuff + (spacing/4), HEIGHT/2-textsize, spacing/2, 2])
            drawtext(scrn, teams[team],
                     [leftbuff + (spacing/2) - (len(str(teams[team]))*textsize*.6)/2,
                      HEIGHT/2-textsize], textsize, textcolor)
            if teams[team] >= winning:
                drawtext(scrn, "Winner",
                         [leftbuff + (spacing/2) - (len("Winner")*textsize*.6)/2,
                          HEIGHT/2+textsize], textsize, (255, 255, 0))

    pyg.display.update()
    clicked = {i : False for i in range(10)}
