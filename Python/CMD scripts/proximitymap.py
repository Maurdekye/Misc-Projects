import pygame as pyg
import sys
import math

def dist(pos1, pos2):
    x, y = pos2[0] - pos1[0], pos2[1] - pos1[1]
    return math.sqrt(x**2 + y**2)

pyg.init()

WIDTH, HEIGHT = 500, 500
scrn = pyg.display.set_mode((WIDTH, HEIGHT))

points = {
    (200, 200) : (255, 0, 0),
    (100, 400) : (0, 255, 0),
    (600, 200) : (0, 0, 255),
    (220, 350) : (255, 0, 255)
    }

while True:
    for e in pyg.event.get():
        if e.type == 12:
            pyg.quit()
            sys.exit()

    scrn.fill((255,255,255))
    i = 0
    for x in range(WIDTH):
        i += 0.5
        for y in range(HEIGHT):
            curdist, curcol = 200, (255, 255, 255)
            for p in points:
                c = points[p]
                d = dist(p, (x, y))*((math.sin(i)/100)+1)
                if d < curdist:
                    curdist = d
                    curcol = c
            scrn.set_at((x, y), curcol)
    pyg.display.update()
        
