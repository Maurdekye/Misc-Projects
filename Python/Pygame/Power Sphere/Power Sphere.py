import pygame as pyg
import sys, math
from random import randrange, random, choice

pyg.init()

scrn = pyg.display.set_mode((800, 600), 0, 32)

clock = 2.0
insize = 50
while True:
    clock += 0.01
    size = min(int(clock), 200)
    print size, clock
    insize = min(max(insize + randrange(-1, 2), size/3), size/2)
    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()

    scrn.fill((0, 0, 0))
    pyg.draw.circle(scrn, (80, 80, 255), [400, 300], int((size-2) + (math.sin(clock) * 2)) )
    pyg.draw.circle(scrn, (130, 130, 240), [400, 300], int(insize))
    dots = []
    lim = 100
    c = 0.0
    for x in xrange(lim):
        c += (math.pi * 2) / lim
        dots += [((math.sin(c) * (random()**0.3) * (size + (size / 9))) + 400,
                  (math.cos(c) * (random()**0.3) * (size + (size / 9))) + 300)]
    for i in xrange(10):
        x, y = choice(dots)
        tx, ty = choice(dots)
        while (tx - x)**2 + (ty - y)**2 > size**2:
            tx, ty = choice(dots)
        pyg.draw.line(scrn, (200, 200, 255), [x, y], [tx, ty])
    pyg.display.update()
