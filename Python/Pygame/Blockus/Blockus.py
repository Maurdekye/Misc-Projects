import pygame as pyg
import sys
from color import *
import blokstruct

xlen = lambda a: range(len(a))

def gclone(grid):
    new = []
    for r in grid:
        new += [list(r)]
    return new

def print_gridlist(gls):
    for grid in gls:
        for row in grid:
            for item in row:
                print(item, end="")
            print()
        print()

def cycleup(ls): return [ls[-1]] + ls[:-1]
def cycledn(ls): return ls[1:] + [ls[0]]

def up_occupants(peices, color=None):
    occ = []
    for p in peices:
        for place in p.occupants():
            if place not in occ:
                if color == None:
                    occ += [place]
                else:
                    if p.color == color:
                        occ += [place]
    return occ

def up_illegals(peices, color):
    illegs = []
    for p in peices:
        for place in p.illegals():
            if place not in illegs and p.color == color: illegs += [place]
    return illegs

def border(width, height):
    bor = [[a, b] for a in [-1, width] for b in range(height)]
    return bor + [[a, b] for b in [-1, height] for a in range(width)]

def pillow(grid):
    frontend = [[0]*len(grid[0])]
    grid = frontend + grid + frontend
    return [[0] + row + [0] for row in grid]

def gridport(filename):
    final = []
    grid = [[]]
    with open(filename+'.txt', 'r') as f:
        for line in f:
            if line == '\n':
                final += [grid]
                grid = [[]]
            else:
                for item in line.split(' '):
                    if item[-1] == '\n':
                        grid[-1] += [item[:-1]]
                        grid += [[]]
                    else:
                        grid[-1] += [item]
    return final
                        
                        

# --- Peice Class --- #

class Peice:
    def __init__(this, color, grid, pos=(0, 0)):
        this.color = color # Int List Length 3
        this.grid = grid # Equal Dimension, 2-Dimensional List
        this.pos = pos # Int List Length 3

    def render(this, surf, scale):
        #blokstruct.printgrid(this.grid)
        for y, row in enumerate(this.grid):
            for x, item in enumerate(row):
                #print x, y
                if item:
                    pos = [(this.pos[i] + c)*scale[i] for i, c in [(0, x), (1, y)]]
                    pyg.draw.rect(surf, this.color, pos + scale)

    def render_border(this, surf, scale):
        for i, p in enumerate(this.grid[0]):
            if p: pyg.draw.rect(surf, white, [(this.pos[0]+i)*scale[0],
                                              this.pos[1]*scale[1] - 1,
                                              scale[0], 3])
            #print (this.pos[0]+i)*scale[0], this.pos[1]*scale[1] - 1, scale[0]

    def occupants(this):
        tags = []
        for y, row in enumerate(this.grid):
            for x, item in enumerate(row):
                if item: tags += [[x+this.pos[0], y+this.pos[1]]]
        return tags
    
    def illegals(this):
        tags = []
        occs = this.occupants()
        for y, row in enumerate(this.grid):
            for x, item in enumerate(row):
                checks = [(x+a, y) for a in [-1, 1]] + [(x, y+a) for a in [-1, 1]]
                for mx, my in checks:
                    if item: tags += [[mx+this.pos[0], my+this.pos[1]]]
        return tags

    def corners(this):
        tags = []
        occs = this.occupants()
        for x, y in occs:
            checks = [[x+a, y+b] for a in [-1, 1] for b in [-1, 1]]
            for mx, my in checks:
                if [mx, my] not in occs and [mx, my] not in tags: tags += [[mx, my]]
        finals = []
        for x, y in tags:
            checks = [(x+a, y) for a in [-1, 1]] + [(x, y+a) for a in [-1, 1]]
            for mx, my in checks:
                if [mx, my] in occs:
                    break
            else:
                finals += [[x, y]]
        return finals

    '''
    def rotate(this):
        if times <= 0: return
        newgrid = [[0 for x in xlen(this.grid[0])] for y in xlen(this.grid)]
        for xr, y in zip(range(len(this.grid[0])), range(len(this.grid))):
            for yr, x in zip(range(len(this.grid))[::-1], range(len(this.grid[0]))):
                newgrid[x][y] = this.grid[xr][yr]
        this.grid = gclone(newgrid)
        this.rotate(times - 1)

    def flip(this):
        newgrid = [[0 for x in xlen(this.grid[0])] for y in xlen(this.grid)]
        for xr, x in zip(range(len(this.grid))[::-1], range(len(this.grid))):
            for yr, y in zip(range(len(this.grid[0])), range(len(this.grid[0]))):
                newgrid[x][y] = this.grid[xr][yr]
        this.grid = gclone(newgrid)
    '''

    def rotate(this):
        this.grid = blokstruct.rotate(this.grid)

    def flip(this):
        this.grid = blokstruct.mirror(this.grid)

    def clone(this):
        return Peice(this.color, gclone(this.grid), this.pos)

# --- Shape Definitons --- #

shapes = [
    
    [[1]],
    
    [[1, 1],
     [0, 0]],
    
    [[1, 1],
     [1, 0]],
    
    [[1, 1],
     [1, 1]],

    [[0, 0, 0],
     [1, 1, 1],
     [0, 0, 0]],
    
    [[0, 1, 1],
     [1, 1, 0],
     [0, 0, 0]],
    
    [[0, 1, 0],
     [1, 1, 1],
     [0, 0, 0]],

    [[1, 0, 0],
     [1, 1, 1],
     [0, 0, 0]],
     
    [[1, 1, 0],
     [1, 1, 1],
     [0, 0, 0]],
     
    [[1, 0, 0],
     [1, 1, 1],
     [0, 1, 0]],
     
    [[1, 1, 0],
     [0, 1, 0],
     [0, 1, 1]],
     
    [[0, 1, 0],
     [1, 1, 1],
     [0, 1, 0]],
     
    [[0, 1, 1],
     [1, 1, 0],
     [1, 0, 0]],
     
    [[1, 1, 1],
     [0, 1, 0],
     [0, 1, 0]],
     
    [[0, 0, 1],
     [0, 0, 1],
     [1, 1, 1]],
     
    [[1, 0, 1],
     [1, 1, 1],
     [0, 0, 0]],

    [[0, 0, 0, 0],
     [1, 1, 1, 1],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],

    [[0, 0, 0, 0],
     [1, 1, 1, 1],
     [0, 0, 1, 0],
     [0, 0, 0, 0]],

    [[0, 0, 0, 0],
     [1, 1, 1, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 0]],

    [[0, 0, 0, 0],
     [1, 1, 1, 0],
     [0, 0, 1, 1],
     [0, 0, 0, 0]],

    [[0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [1, 1, 1, 1, 1],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]],
]

#shapes = gridport('grid5')

# --- Initializations --- #

pyg.init()

WID, HIH = 1080, 1080
DIM = [WID, HIH]
clock = 0
clicktime = 0
horigrid = 14
vertgrid = 14

scrn = pyg.display.set_mode(DIM, pyg.FULLSCREEN)

gheight = HIH / vertgrid
gwidth = WID / horigrid
gscale = [gwidth, gheight]

teams = [
    orange,
    blue
    ][:4]

cur = teams[0]

waiting = {color : [Peice(color, grid) for grid in shapes] for color in teams}
placed = [Peice(grey, [[1]], (a, b)) for a, b in [(-1, -1), (horigrid, vertgrid), (-1, vertgrid), (horigrid, -1)]]
for i, c in enumerate(teams): placed[i].color = c
occupants = up_occupants(placed)
team_occupants = {color : up_occupants(placed, color) for color in teams}
illegals = {color : up_illegals(placed, color) for color in teams}

DEBUG = True

# --- Main Loop --- #

while True:
    clock += 1

    mx, my = pyg.mouse.get_pos()
    
    gridx = mx - (mx % gwidth)
    gridy = my - (my % gheight)
    for i, p in enumerate(waiting[cur]):
        waiting[cur][i].pos = [int(mx / float(gwidth)) - len(waiting[cur][-1].grid) / 2,
                                int(my / float(gheight)) - len(waiting[cur][-1].grid) / 2]
    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()
        if ev.type == 5:
            if ev.button in [4, 5]:
                waiting[cur] = {4:cycleup(waiting[cur]), 5:cycledn(waiting[cur])}[ev.button]
            else:
                if ev.button == 1:
                    for p in occupants + illegals[cur] + border(horigrid, vertgrid):
                        if p in waiting[cur][-1].occupants():
                            print("Nope.")
                            if DEBUG: print("Failed general check")
                            break
                    else:
                        if DEBUG: print(team_occupants[cur])
                        for c in waiting[cur][-1].corners():
                            if c in team_occupants[cur]:
                                # Block entered when peice is in legal position
                                placed += [waiting[cur].pop()]
                                if waiting[cur] == []: waiting[cur] = [Peice(black, [[0]])]
                                occupants = up_occupants(placed)
                                team_occupants[cur] = up_occupants(placed, cur)
                                illegals[cur] = up_illegals(placed, cur)
                                cur = teams[(teams.index(cur) + 1) % len(teams)]
                                break
                        else:
                            print("Nope.")
                            if DEBUG: print("Failed corner check")
                if ev.button == 2:
                    waiting[cur][-1].flip()
                if ev.button == 3:
                    waiting[cur][-1].rotate()
        if ev.type == 2:
            if ev.key == 114:
                waiting[cur][-1].rotate()
            elif ev.key == 102:
                waiting[cur][-1].flip()
            elif ev.key == 273:
                waiting[cur] = cycleup(waiting[cur])
            elif ev.key == 274:
                waiting[cur] = cycledn(waiting[cur])
            elif ev.key == 27:
                pyg.quit()
                sys.exit()
    
    scrn.fill(grey)

    # - Debug Renders - #
    
    if DEBUG:
        for i in illegals[cur]:
            pyg.draw.rect(scrn, green, [i[0]*gwidth, i[1]*gheight, gwidth, gheight] )
        for o in occupants:
            pyg.draw.rect(scrn, yellow, [o[0]*gwidth, o[1]*gheight, gwidth, gheight] )
        for c in waiting[cur][-1].corners():
            pyg.draw.rect(scrn, blue, [c[0]*gwidth, c[1]*gheight, gwidth, gheight] )

    # - Polymino Renders - #
    
    for p in placed:
        p.render(scrn, gscale)
    waiting[cur][-1].render(scrn, gscale)

    # - Grid Renders - #
    
    for i in range(horigrid+1):
        pyg.draw.rect(scrn, dark, [i*gwidth - 1, 0, 3, HIH])
    for i in range(vertgrid+1):
        pyg.draw.rect(scrn, dark, [0, i*gheight - 1, WID, 3])

    # - Placing Border Render - #

    #waiting[cur][-1].render_border(scrn, gscale)
        
    pyg.display.update()
