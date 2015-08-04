import pygame as pyg
import sys
import math
from calculus import *

pyg.init()

# --- Variable Definitions --- #

WIDTH, HEIGHT = 800, 800
DIM = min(WIDTH, HEIGHT)
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
scale_h = [-8.0, 8.0]
scale_v = [-8.0, 8.0]
lastStroke = (WIDTH / 2, HEIGHT / 2)
request = False

default = scale_h, scale_v

# --- Function Definitons --- #

def drawtext(screen, text, pos, color=(0,0,0), size=25):
    screen.blit(pyg.font.SysFont("monospace", int(size)).render(str(text), 0, color), pos)

def scale_down(scale, compare, num):
    num = ( float(num) / compare ) * float(abs(scale[0]-scale[1]))
    num += scale[0]
    return num

def scale_up(scale, compare, num):
    num -= scale[0]
    num = ( float(num) / float(abs(scale[0]-scale[1])) ) * compare
    return int(num)

def clamp(mn, v, mx):
    return max(mn, min(v, mx))

def freerange(low, hih=None, delim=1):
    if hih == None:
        low, hih = 0, low
    i = low
    while i < hih:
        yield i
        i += delim

# --- Graphing Equations --- #

def abs_fractal(x, depth=20):
    delimit = 4.0
    for i in range(depth):
        x = abs(x)-delimit
        delimit /= 4.0
    return x

f = lambda x: (5*x**2 + 3*x) / (x**3)
g = lambda x: f(-x)
s = lambda x: x**2
d = indefinite_integral(s, 0)

equations = [
    s,
    d
    ]


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
                scale_h, scale_v = default
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
    scrn.fill((200, 200, 255))

    # -- Draw Axis
    '''
    intersperse = 2**int(math.log(srange[0], 2))/8.0
    if request:
        print 'axis info'
        print 'start:\t\t\t', int(scale_h[0])
        print 'end:\t\t\t', int(scale_h[1]) + 1
        print 'integral:\t\t', intersperse
        print 'loop progression:\t',
    for i in freerange(int(scale_h[0]), int(scale_h[1])+1, intersperse):
        if request: print i,
        y = scale_up(scale_h, WIDTH, i)
        pyg.draw.line(scrn, (190, 190, 190), (WIDTH - y, 0), (WIDTH - y, HEIGHT))
    if request: print
    intersperse = 2**int(math.log(srange[1], 2))/8.0
    
    for i in freerange(int(scale_v[0]), int(scale_v[1])+1, intersperse):
        x = scale_up(scale_v, HEIGHT, i)
        pyg.draw.line(scrn, (190, 190, 190), (0, HEIGHT - x), (WIDTH, HEIGHT - x))
    '''
    y = scale_up(scale_h, WIDTH, 0)
    x = scale_up(scale_v, HEIGHT, 0)
    pyg.draw.line(scrn, (0, 0, 200), (0, HEIGHT - x), (WIDTH, HEIGHT - x))
    pyg.draw.line(scrn, (0, 0, 200), (WIDTH - y, 0), (WIDTH - y, HEIGHT))
    
    # -- Calculate Equations
    plots = {}
    eqplots = [0 for x in range(len(equations))]
    for x in range(WIDTH):
        gx = scale_down(scale_h, WIDTH, x)
        for i, f in enumerate(equations):
            try: gy = f(gx)
            except:
                try: gy = f(gx + 0.0000001)
                except: gy = None
            if WIDTH-x == mpos[0]:
                try: eqplots[i] = round(gy, 3)
                except: eqplots[i] = 'undefined'
            if gy != None:
                y = scale_up(scale_v, HEIGHT, gy)
                try: plots[f] += [(x, y)]
                except KeyError: plots[f] = [(x, y)]
    eqplots = eqplots[::-1]
                
    # -- Draw Equations
    text = 'x = ' + str(round(scale_down(scale_h, WIDTH, mpos[0]), 3))
    drawtext(scrn, text, [10, HEIGHT - (DIM/32) - 10], size=DIM/32)
    for c, f in enumerate(plots):
        drawtext(scrn, 'Eq. {}: {}'.format(c+1, eqplots[c]), [5, (DIM/32) * c], colors[c%len(colors)], DIM/32)
        for i in range(len(plots[f]) - 1):
            x, y = plots[f][i]
            nx, ny = plots[f][i+1]
            if x > WIDTH or x < 0: continue
            if y > HEIGHT*2 or y < 0: continue
            if abs(nx-x) > 2: continue
            pyg.draw.aaline(scrn, colors[c%len(colors)], (WIDTH - x, HEIGHT - y), (WIDTH - nx, HEIGHT - ny))
        pyg.draw.circle(scrn, colors[c%len(colors)], [mpos[0], clamp(0, HEIGHT - scale_up(scale_v, HEIGHT, eqplots[c]), HEIGHT)], 3)
    
    pyg.display.update()
    request = False
