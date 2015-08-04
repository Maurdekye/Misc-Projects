import pygame as pyg
import sys
import math
from basefiles import *

screen = pyg.display.set_mode((SIZE[0], SIZE[1]), 0, 32)

points = []

mpoint = Point(0, 0)
apoint = Point(SIZE[0]/2, SIZE[1]/2)
moveline = Line(mpoint, apoint)
##fidel = 10
while True:
    lines = [Line(*points[x:x+2]) for x in xrange(0, len(points) - (len(points) % 2), 2)]
    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()
        if ev.type == 5:
            if ev.button == 1:
                points += [Point(*pyg.mouse.get_pos())]
            if ev.button == 2:
                apoint.x, apoint.y = pyg.mouse.get_pos()
            if ev.button == 3:
                if not points == []: del points[-1]
    mpoint.x, mpoint.y = pyg.mouse.get_pos()
    screen.fill(black)
    for p in points: p.render(screen)
    mpoint.render(screen)
    apoint.render(screen)
    moveline.render(screen)
    for l in lines:
        if moveline.intersects(l):
            l.render(screen, red)
            ipt = moveline.intersection(l)
            pyg.draw.rect(screen, red, [int(ipt.x)-1, int(ipt.y)-1, 3, 3])
            ipt.render(screen, blue)
        else:
            l.render(screen, grey)
##    for x in xrange(0, SIZE[0], fidel):
##        for y in xrange(0, SIZE[1], fidel):
##            cp = Point(x, y)
##            c = grey
##            if Line(mpoint, cp).intersects(lines[0]):
##                if lines[0].side(cp):
##                    c = red
##                elif not lines[0].side(cp):
##                    c = red
##            #c = {1:red, 0:grey}[lines[0].side(Point(x, y)) and lines[0].inbox(Point(x, y))]
##            pyg.draw.rect(screen, c, (x, y, fidel, fidel))
##    lines[0].render(screen, black)
    pyg.display.update()
