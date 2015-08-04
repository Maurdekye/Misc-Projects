import pygame as pyg
import sys
from random import randint as rand
from math import sin, cos, pi, sqrt

class Stick(object):
    def __init__(this, pos1, pos2, color):
        this.pos1 = pos1
        this.pos2 = pos2
        this.color = color

    def draw(this, screen):
        pyg.draw.line(screen, this.color, this.pos1, this.pos2, 1)

    def __str__(this):
        return "{} to {}; RGB is {}.".format(this.pos1, this.pos2, this.color)

    def get_length(this):
        return sqrt( (this.pos2[0] - this.pos1[0])**2 + (this.pos1[1] - this.pos2[1])**2 )

class Timer(object):
    def __init__(this, timings, events):
        this.timings = timings
        this.events = events
        this.time = 0
        this.activeEvent = 0

    def advance(this):
        if this.activeEvent < len(this.timings): this.time += 1
        if this.time > this.timings[this.activeEvent % len(this.timings)]:
            this.time = 0
            this.activeEvent += 1
            this.update()

    def update(this):
        this.events[(this.activeEvent - 1) % len(this.timings)]()
        
def rand_stick(bound1, bound2, length):
    s = lambda x:raw_input(x)
    l = float(length) / 2
    bound1, bound2 = (min(bound1[0], bound2[0]), min(bound1[1], bound2[1])), (max(bound1[0], bound2[0]), max(bound1[1], bound2[1]))
    bound1 = (bound1[0] + length, bound1[1] + length)
    bound2 = (bound2[0] - length, bound2[1] - length)
    cent = (rand(bound1[0], bound2[0]), rand(bound1[1], bound2[1]))
    ang = rand(1, 100000000) / 100000000.0
    ang *= pi
    point1 = cent[0] + (sin(ang) * l), cent[1] + (cos(ang) * l)
    point2 = cent[0] + (sin(ang + pi) * l), cent[1] + (cos(ang + pi) * l)
    return Stick(point1, point2, (0, 0, 0))

def drawtext(text, pos):
    scrn.blit(pyg.font.SysFont("monospace", 40).render(str(text), 0, (255,0,0)), pos)

width, height = 800, 800
spacing = 15
numsticks = 4000

pyg.init()

scrn = pyg.display.set_mode((width, height), 0, 32)
bg = pyg.image.load("bg.jpg").convert()

sticks = []
horidraws = [0]
topcounter = [0, 0]
adv = [False]
fin = [False, 0]

def assign_stick():
    if sticks[-1].pos1[1] // spacing != sticks[-1].pos2[1] // spacing:
        topcounter[0] += 1
    else: topcounter[1] += 1
    sticks.pop()

def finish():
    adv = [False]
    fin[0] = True

timing, events = [], []

timing += [1 for x in xrange(numsticks)]
events += [lambda : sticks.append(rand_stick((0, 0), (width, height), spacing)) for x in xrange(numsticks)]

def f(): horidraws[0] += 1
timing += [60 for x in xrange(height // (spacing * 2))] + [800]
events += [f for x in xrange(height // (spacing * 2))] + [lambda : 0]

timing += [1 for x in xrange(numsticks)] + [800]
events += [assign_stick for x in xrange(numsticks)] + [lambda : 0]

def step1():adv[0] = [True]
timing += [900, 900]
events += [step1, finish]

tempo = Timer(timing, events)

"""
for i in xrange(20):
    newstick = rand_stick((0, 0), (width, height), spacing)
    sticks.append(newstick)
    print newstick.get_length(), "({}, {})".format(newstick.pos1[1] // spacing, newstick.pos2[1] // spacing)
"""

while True:
    tempo.advance()
    fin[1] = float(topcounter[0] + 1) / float(topcounter[1] + 1)
    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()
    
    scrn.fill((255, 255, 255))
    for s in sticks: s.draw(scrn)
    for x in xrange(horidraws[0]): pyg.draw.line(scrn, (200, 200, 200), (0, x * (spacing * 2)), (width, x * (spacing * 2)), 1)
    if not fin[0]:
        for x in xrange(2):
            if not adv[0]:
                if topcounter[x] != 0: drawtext(topcounter[x], ( 40 + (x * 650), 40 ))
            else:
                if topcounter[x] != 0: drawtext(topcounter[x], ( 350, 155 + (x * 45) ))
    if fin[0]: drawtext(fin[1], (350, 245))
    if adv[0] and not fin[0]: pyg.draw.line(scrn, (0, 0, 0), (350, 202),
                             ( 350 + (max(len(str(topcounter[0])), len(str(topcounter[1])) ) - 1) * 40, 202), 2)
    pyg.display.update()
