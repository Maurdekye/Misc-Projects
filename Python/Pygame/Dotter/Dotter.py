import pygame as pyg
import sys
import math
import random
from calculus import *

pyg.init()

# --- Variable Definitions --- #

WIDTH, HEIGHT = 800, 800
scrn = pyg.display.set_mode((WIDTH, HEIGHT))

colors = [
    (196, 0, 0),
    (0, 196, 0),
    (0, 0, 196),
    (255, 255, 0),
    (0, 255, 255),
    (0, 0, 0)
    ]
zoom_factor = 1.125
pressing = {x : False for x in range(1, 10)}
scale_h = [-3, 3]
scale_v = [-3, 3]
lastStroke = (WIDTH / 2, HEIGHT / 2)
request = False
scalefactor = 10

# --- Function Definitons --- #

def scale_down(scale, compare, num):
    num = ( float(num) / compare ) * float(abs(scale[0]-scale[1]))
    num += scale[0]
    return num

def scale_up(scale, compare, num):
    num -= scale[0]
    num = ( float(num) / float(abs(scale[0]-scale[1])) ) * compare
    return int(num)

def span(lst):
    return abs(lst[1]-lst[0])

def freerange(low, hih=None, delim=1):
    if hih == None:
        low, hih = 0, low
    i = low
    while i < hih:
        yield i
        i += delim

def getSamples(size, rangex, rangey):
    return [[(random.random() * span(rangex)) - abs(rangex[0]),
                (random.random() * span(rangey)) - abs(rangey[0])] for s in range(size)]

# --- Graphing Equations --- #

def abs_fractal(x, depth=20):
    delimit = 4.0
    for i in range(depth):
        x = abs(x)-delimit
        delimit /= 4.0
    return x

def xpos(rand, dots):
    return math.log(rand)

def ypos(rand, dots):
    return rand**2

size = 10000
rangex = [-2, 2]
rangey = [-2, 2]

samples = getSamples(size, rangex, rangey)

# --- Main Loop --- #

while True:
    srange = [scale_h[1] - scale_h[0], scale_v[1] - scale_v[0]]
    mpos = pyg.mouse.get_pos()
    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()
        if ev.type == 2:
            if ev.key == 27:
                pyg.quit()
                sys.exit()
            if ev.key == 114:
                request = True
            if ev.key == 32:
                samples = getSamples(size, rangex, rangey)
            if ev.key == 273:
                size += scalefactor
                print(size)
                samples = getSamples(size, rangex, rangey)
            if ev.key == 274:
                size = max(1, size - scalefactor)
                print(size)
                samples = getSamples(size, rangex, rangey)
        if ev.type == 5: pressing[ev.button] = True
        if ev.type == 6: pressing[ev.button] = False
        if ev.type == 5:

            # -- Zooming
            adds = [scale_down(s, c, v) for s, c, v in zip([scale_h, scale_v], [WIDTH, HEIGHT], mpos)]
            print(adds)
            print(scale_down(scale_h, WIDTH, mpos[0]))
            adds = [0, 0]
            if ev.button == 5:
                scale_h = [a * zoom_factor - adds[0] for a in scale_h]
                scale_v = [a * zoom_factor - adds[1] for a in scale_v]
            if ev.button == 4:
                scale_h = [a / zoom_factor - adds[0] for a in scale_h]
                scale_v = [a / zoom_factor - adds[1] for a in scale_v]

    # -- Dragging
    if pressing[1]:
        delta = [a - b for a, b in zip(mpos, lastStroke)]
        dscale = [scale_down(s, c, v) - s[0] for s, c, v in zip([scale_h, scale_v], [WIDTH, HEIGHT], delta)]
        scale_h = [v + dscale[0] for v in scale_h]
        scale_v = [v + dscale[1] for v in scale_v]
    lastStroke = mpos
    if request:
        print('X', scale_h)
        print('Y', scale_v)

    scrn.fill((200, 200, 255))

    # -- Draw Axis
    y = scale_up(scale_h, WIDTH, 0)
    x = scale_up(scale_v, HEIGHT, 0)
    pyg.draw.line(scrn, (0, 0, 200), (0, HEIGHT - x), (WIDTH, HEIGHT - x))
    pyg.draw.line(scrn, (0, 0, 200), (WIDTH - y, 0), (WIDTH - y, HEIGHT))
    
    # -- Calculate Equations
    points = []
    for rx, ry in samples:
        gx = 0
        try: gx = xpos(rx, points)
        except: pass
        gy = 0
        try: gy = ypos(ry, points)
        except: pass
        points += [[gx, gy]]
    
    # -- Draw Equations
    for x, y in points:
        sx, sy = scale_up(scale_h, WIDTH, x), scale_up(scale_v, HEIGHT, y)
        if request: print('-', x, y, ' | ', sx, sy)
        if sx > WIDTH or sx < 0: continue
        if sy > HEIGHT or sy < 0: continue
        pyg.draw.rect(scrn, (0, 0, 0), [WIDTH-sx-1, WIDTH-sy-1, 3, 3])
        
    pyg.display.update()
    request = False
