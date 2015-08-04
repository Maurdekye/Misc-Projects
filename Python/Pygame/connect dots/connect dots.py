import pygame as pyg
import sys
import copy

def rlen(iterable): return range(len(iterable))
def xrlen(iterable): return xrange(len(iterable))

def drawtext(screen, text, pos, size=25, color=(0, 0, 0)):
    screen.blit(pyg.font.SysFont("monospace", size).render(str(text), 0, color), pos)
    
def trim(txt):
    for l in txt:
        if l not in [' ', '\n']: break
    else: return ''
    for i in xrlen(txt):
        if txt[i] not in [' ', '\n']:
            txt = txt[i:]
            break
    for i in rlen(txt)[::-1]:
        if txt[i] not in [' ', '\n']:
            txt = txt[:i+1]
            break
    return txt

def checkfor(lis, lisof):
    for l in lisof:
        if l == lis or l[::-1] == lis:
            return True
    return False

'''
p1 = ''
while p1 == '':
    p1 = trim(raw_input("Please enter the inital of player 1. ")).lower()
p1 = p1[0]
p2 = ''
while p2 == '':
    p2 = trim(raw_input("Please enter the inital of player 2. ")).lower()
p2 = p2[0]
'''

pyg.init()

pressing = {i : False for i in xrange(512)}
clock = 0
dist = 40
turn = True
WIDTH, HEIGHT = 800, 800
WIDTH = WIDTH - WIDTH % dist
HEIGHT = HEIGHT - HEIGHT % dist
scrn = pyg.display.set_mode((WIDTH, HEIGHT))
lines = []
fills = []
limit = (WIDTH / dist - 2) * (HEIGHT / dist - 1)
limit += (HEIGHT / dist - 2) * (WIDTH / dist - 1)

firstPos = None

while True:
    clock += 1
    mse_pos = pyg.mouse.get_pos()
    mod_pos = [(p + dist/2) - ((p + dist/2) % dist) for p in mse_pos]
    grd_pos = [p / dist for p in mod_pos]
    
    select = False

    # --- Rendering
    scrn.fill((255, 255, 255))
    for x in xrange(dist, WIDTH, dist):
        for y in xrange(dist, HEIGHT, dist):
            if grd_pos == [x/dist, y/dist]:
                select = True
                pyg.draw.circle(scrn, (255, 0, 0), [x, y], 7)
            pyg.draw.circle(scrn, (0, 0, 0), [x, y], 5)
    for l in lines:
        pyg.draw.line(scrn, (0, 0, 0), [p*dist for p in l[0]], [p*dist for p in l[1]], 3)
    if firstPos != None:
        pyg.draw.circle(scrn, (0, 255, 0), [p*dist for p in firstPos], 5)
    pyg.display.update()

    # --- Event Tracking
    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()
        if ev.type == 2:
            if ev.key == 27:
                pyg.quit()
                sys.exit()
            pressing[ev.key] == True
        if ev.type == 3: pressing[ev.key] == False
        if ev.type == 5:
            if ev.button == 1:
                if select and len(lines) < limit:
                    if firstPos == None:
                        firstPos = list(grd_pos)
                    else:
                        adds = [
                            [1, 0],
                            [0, 1],
                            [-1, 0],
                            [0, -1]
                            ]
                        goods = [[firstPos[0]+a, firstPos[1]+b] for a, b in adds]
                        if grd_pos in goods:
                            possible = [firstPos, grd_pos]
                            if possible[::-1] not in lines and possible not in lines:
                                lines += [possible]
                                firstPos = None
                                turn = not turn

                                # --- Check for Scoring
                                a = possible
                                if a[0][0] > a[1][0]: a[0][0], a[1][0] = a[1][0], a[0][0]
                                if a[0][1] > a[1][1]: a[1][1], a[0][1] = a[0][1], a[1][1]
                                print "PSB", possible
                                new = copy.deepcopy(possible)
                                if possible[0][0] != possible[1][0]:
                                    new[0][1] += 1
                                    new[1][1] += 1
                                    print "A", new
                                    if not checkfor(new, lines):
                                        lines += [copy.deepcopy(new)]
                                        firstPos = None
                                        turn = not turn
                                        new[0][1] -= 1
                                        new[0][0] += 1
                                        if not checkfor(new, lines):
                                            lines += [copy.deepcopy(new)]
                                            firstPos = None
                                            turn = not turn
                                        new[0][0] -= 1
                                        new[1][0] -= 1
                                        if not checkfor(new, lines):
                                            lines += [copy.deepcopy(new)]
                                            firstPos = None
                                            turn = not turn
                                    else:
                                        new[0][1] -= 2
                                        new[1][1] -= 2
                                        print "B", new
                                        if not checkfor(new, lines):
                                            lines += [copy.deepcopy(new)]
                                            firstPos = None
                                            turn = not turn
                                            new[0][1] -= 2
                                            new[0][0] += 1
                                            if not checkfor(new, lines):
                                                lines += [copy.deepcopy(new)]
                                                firstPos = None
                                                turn = not turn
                                            new[0][0] -= 2
                                            new[1][0] -= 2
                                            if not checkfor(new, lines):
                                                lines += [copy.deepcopy(new)]
                                                firstPos = None
                                                turn = not turn
                                else:
                                    new[0][0] += 1
                                    new[1][0] += 1
                                    print "C", new
                                    if not checkfor(new, lines):
                                        lines += [copy.deepcopy(new)]
                                        firstPos = None
                                        turn = not turn
                                    else:
                                        new[0][0] -= 2
                                        new[1][0] -= 2
                                        print "D", new
                                        if not checkfor(new, lines):
                                            lines += [copy.deepcopy(new)]
                                            firstPos = None
                                            turn = not turn
                            else:
                                firstPos = None
                        else:
                            firstPos = None
            pressing[ev.button] == True
        if ev.type == 6: pressing[ev.button] == False


