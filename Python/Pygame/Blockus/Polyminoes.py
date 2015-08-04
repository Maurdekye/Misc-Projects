import pygame as pyg
import sys
from color import *

xlen = lambda a: xrange(len(a))

def gclone(grid):
    new = []
    for r in grid:
        new += [list(r)]
    return new

def print_gridlist(gls):
    for grid in gls:
        for row in grid:
            for item in row:
                print item,
            print
        print

def cycleup(ls): return [ls[-1]] + ls[:-1]
def cycledn(ls): return ls[1:] + [ls[0]]

def up_occupants(peices, color):
    occ = []
    for p in peices:
        for place in p.occupants():
            if place not in occ and p.color == color: occ += [place]
    return occ

def up_illegals(peices, color):
    illegs = []
    for p in peices:
        for place in p.illegals():
            if place not in illegs and p.color == color: illegs += [place]
    return illegs

def border(width, height):
    bor = [[a, b] for a in [-1, width] for b in xrange(height)]
    return bor + [[a, b] for b in [-1, height] for a in xrange(width)]

def pillow(grid):
    frontend = [[0]*len(grid[0])]
    grid = frontend + grid + frontend
    return [[0] + row + [0] for row in grid]

# --- Peice Class --- #

class Peice:
    def __init__(this, color, grid, pos=(0, 0)):
        this.color = color # Int List Length 3
        this.grid = grid # Equal Dimension, 2-Dimensional List
        this.pos = pos # Int List Length 3

    def render(this, surf, scale):
        for y in xlen(this.grid):
            for x in xlen(this.grid[0]):
                if this.grid[x][y]:
                    pos = [(this.pos[i] + c)*scale[i] for i, c in [(0, x), (1, y)]]
                    pyg.draw.rect(surf, this.color, pos + scale)

    def occupants(this):
        tags = []
        for y, row in enumerate(this.grid):
            for x, item in enumerate(row):
                if item: tags += [[y+this.pos[0], x+this.pos[1]]]
        return tags
    
    def illegals(this):
        tags = []
        occs = this.occupants()
        for y, row in enumerate(this.grid):
            for x, item in enumerate(row):
                checks = [(x+a, y) for a in [-1, 1]] + [(x, y+a) for a in [-1, 1]]
                for mx, my in checks:
                    if item: tags += [[my+this.pos[0], mx+this.pos[1]]]
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
                    
        '''
                s_checks = [(mx+a, my) for a in [-1, 1]] + [(mx, my+a) for a in [-1, 1]]
                for sx, sy in s_checks:
                    if not [sx, sy] in occs:
                        tags += [[mx-this.pos[1], my-this.pos[0]]]
        '''
        return finals
             
    def rotate(this, times=1):
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

# --- Shape Definitons --- #

shapes = [grid for grid in [
    
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
]]

pyg.init()

WID, HIH = 800, 800
DIM = [WID, HIH]
clock = 0
clicktime = 0
vertgrid = 20
horigrid = 20

scrn = pyg.display.set_mode(DIM, 0, 32)

gheight = HIH / vertgrid
gwidth = WID / horigrid
gscale = [gwidth, gheight]

waiting = [Peice(red, grid) for grid in shapes]
placed = [Peice(red, [[1]], (a, b)) for a in [-1, horigrid] for b in [-1, vertgrid]]
occupants, illegals = up_occupants(placed, red), up_illegals(placed, red)
DEBUG = False

# --- Main Loop --- #

while True:
    clock += 1

    mx, my = pyg.mouse.get_pos()
    
    gridx = mx - (mx % gwidth)
    gridy = my - (my % gheight)

    if waiting != []:
        waiting[-1].pos = [int(mx / float(gwidth)) - len(waiting[-1].grid) / 2,
                           int(my / float(gheight)) - len(waiting[-1].grid) / 2]
    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()
        if ev.type == 5:
            if ev.button in [4, 5]:
                waiting = {4:cycleup(waiting), 5:cycledn(waiting)}[ev.button]
            else:
                bad = False
                for p in occupants + illegals + border(horigrid, vertgrid):
                    for c in waiting[-1].occupants():
                        if p == c:
                            bad = True
                            print "Nope."
                            break
                    if bad: break
                else:
                    for c in waiting[-1].corners():
                        if c in occupants:
                            placed += [waiting.pop()]
                            if waiting == []: waiting = [Peice(black, [[0]])]
                            occupants = up_occupants(placed, red)
                            illegals = up_illegals(placed, red)
                            break
                    else:
                        print "Nope."
        if ev.type == 2:
            if ev.key == 114:
                waiting[-1].rotate()
            elif ev.key == 102:
                waiting[-1].flip()
    
    scrn.fill(grey)
    
    if DEBUG:
        for i in illegals:
            pyg.draw.rect(scrn, green, [i[0]*gwidth, i[1]*gheight, gwidth, gheight] )
        for o in occupants:
            pyg.draw.rect(scrn, yellow, [o[0]*gwidth, o[1]*gheight, gwidth, gheight] )
        for c in waiting[-1].corners():
            pyg.draw.rect(scrn, blue, [c[0]*gwidth, c[1]*gheight, gwidth, gheight] )
            
    for p in placed:
        p.render(scrn, gscale)
    if waiting != []:
        waiting[-1].render(scrn, gscale)

    for i in xrange(horigrid+1):
        pyg.draw.rect(scrn, dark, [i*gwidth - 1, 0, 3, HIH])
    for i in xrange(vertgrid+1):
        pyg.draw.rect(scrn, dark, [0, i*gheight - 1, WID, 3])
        
    pyg.display.update()
