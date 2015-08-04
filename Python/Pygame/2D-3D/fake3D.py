import pygame as pyg
import sys
import math
from basefiles import *

SIZE = Point(800, 800)
pressing = {x : 0 for x in xrange(1000)}

pyg.init()

screen = pyg.display.set_mode((SIZE.x, SIZE.y), 0, 32)

polys = 16
inc = math.pi / (polys / 2)
cpos = 200, 300
radi = 50
colpoints = [Point(math.sin(x * inc), math.cos(x * inc)) for x in xrange(polys + 1)]
colpoints = [Point(p.x * radi + cpos[0], p.y * radi + cpos[1]) for p in colpoints]
column = [Line(p, colpoints[i+1]) for i, p in enumerate(colpoints[:-1])]

renders = [
    Line(Point(300, 200), Point(500, 600)),
    Line(Point(100, 20), Point(20, 100)),
    Line(Point(700, 20), Point(780, 100)),
    Line(Point(500, 300), Point(75000, 320))
    ] + column

cam = Camera(Point(SIZE.x/2, SIZE.y/2), 0, renders)

while True:
    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()
        if ev.type == 2:
            pressing[ev.key] = 1
        if ev.type == 3:
            pressing[ev.key] = 0
    
    cam.pos.x += cam.getsin() * pressing[273]
    cam.pos.y += cam.getcos() * pressing[273]
    cam.ang += pressing[275]
    cam.ang -= pressing[276]
    cam.pos.x = min(max(30, cam.pos.x), SIZE.x - 30)
    cam.pos.y = min(max(30, cam.pos.y), SIZE.y - 30)

    screen.fill(black)
    cam.render3D(screen, SIZE)
    pyg.display.update()
