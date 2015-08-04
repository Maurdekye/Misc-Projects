import pygame as pyg
import sys
import math
import random

class Point:
    def __init__(this, x, y):
        this.x = x
        this.y = y

    def __str__(this):
        return '({}, {})'.format(this.x, this.y)

class Line:
    def __init__(this, a, dist, deg):
        this.a = a
        this.dist = dist
        this.deg = deg
        rad = (deg/180.0) * math.pi
        newx = this.a.x + math.sin(rad)*dist
        newy = this.a.y + math.cos(rad)*dist
        this.b = Point(newx, newy)

    def render(this, scrn):
        pyg.draw.line(scrn, (0, 0, 0), [this.a.x, this.a.y], [this.b.x, this.b.y])

    def __str__(this):
        return "{} to {}".format(this.a, this.b)

def update():
    new = []
    old = [Line(Point(10.0, 10.0), length, 0)]
    lines = []
    for i in range(25):
        for l in old:
            new += [Line(l.b, length, l.deg)]
            new += [Line(l.b, length, l.deg + 45/(2**i))]
        lines += new
        old = list(new)
        new = []
    return lines
    

pyg.init()

errat = 40
length = 30
pressing = {i : False for i in range(512)}
clock = 0
WIDTH, HEIGHT = 800, 800
scrn = pyg.display.set_mode((WIDTH, HEIGHT))

lines = update()

while True:
    clock += 1
    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()
        if ev.type == 2:
            if ev.key == 27:
                pyg.quit()
                sys.exit()
            if ev.key == 32:
                lines = update()
                '''
                errat = min(errat+1, 180)
                print errat
                lines = [Line(Point(400.0, 0.0), length, random.randrange(-errat, errat))]
                while lines[-1].b.y < HEIGHT:
                    if lines[-1].b.y < length:
                        lines += [Line(lines[-1].b, length, random.randrange(max(-errat, -89), min(errat, 89)))]
                    elif lines[-1].b.x > WIDTH - length:
                        lines += [Line(lines[-1].b, length, random.randrange(-errat, 0))]
                    elif lines[-1].b.x < length:
                        lines += [Line(lines[-1].b, length, random.randrange(0, errat))]
                    else:
                        lines += [Line(lines[-1].b, length, random.randrange(-errat, errat))]
                print len(lines), 'lines'
                '''
            pressing[ev.key] == True
        if ev.type == 3: pressing[ev.key] == False
        if ev.type == 5: pressing[ev.button] == True
        if ev.type == 6: pressing[ev.button] == False
        
    scrn.fill((255, 255, 255))
    for l in lines:
        l.render(scrn);
    pyg.display.update()
