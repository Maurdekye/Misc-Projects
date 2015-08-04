import pygame as pyg
import sys
import math
from random import randrange as rnd

def dist(pos1, pos2):
    x, y = pos2[0] - pos1[0], pos2[1] - pos1[1]
    return math.sqrt(x**2 + y**2)

pyg.init()

WIDTH, HEIGHT = 500, 500
scrn = pyg.display.set_mode((WIDTH, HEIGHT))

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (0, 0, 0),
    (128, 128, 128)
    ]

points = {(rnd(WIDTH), rnd(HEIGHT)) : c for c in colors}

pixels = {(x, y) : (255, 255, 255) for x in range(WIDTH) for y in range(HEIGHT)}

while True:
    for e in pyg.event.get():
        if e.type == 12:
            pyg.quit()
            sys.exit()

    scrn.fill((255,255,255))
    for x in range(WIDTH):
        for y in range(HEIGHT):
            curdist, curcol = 100, (255, 255, 255)
            for p in points:
                c = points[p]
                d = dist(p, (x, y))
                if d < curdist:
                    curdist = d
                    curcol = c
            pixels[(x, y)] = curcol
            for xa, ya in [[x, y-1], [x, y+1], [x-1, y], [x+1, y]]:
                if (xa, ya) in pixels and pixels[(xa, ya)] != curcol:
                    scrn.set_at((x, y), (0, 0, 0))
                    break
            else:
                scrn.set_at((x, y), curcol)
    pyg.display.update()
        
