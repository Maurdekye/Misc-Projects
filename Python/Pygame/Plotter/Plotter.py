import pygame as pyg
import sys
import math
import mandelbrot
import calculus

pyg.init()
WIDTH, HEIGHT = 600, 600
scrn = pyg.display.set_mode((WIDTH, HEIGHT))

pressing = {x : False for x in xrange(1, 10)}
scale_h = [-16, 16]
scale_v = [-16, 16]
scale_zoom = 1.25
translate = 0.25
lastStroke = (WIDTH / 2, HEIGHT / 2)

anchor = [0, 0]

def scale_down(scale, compare, num):
    num = ( float(num) / compare ) * float(abs(scale[0]-scale[1]))
    num += scale[0]
    return num

def scale_up(scale, compare, num):
    num -= scale[0]
    num = ( float(num) / float(abs(scale[0]-scale[1])) ) * compare
    return int(num)

formR = lambda x, y: mandelbrot.mandelbrot(x, y, x, y, 100)
formG = lambda x, y: 0
formB = lambda x, y: 0

def upgrid(wid, high):
    percent, lastpercent = 0.0, 0.0
    print 'Processing...'
    new = [[[] for x in xrange(WIDTH)] for y in xrange(HEIGHT)]
    mx, my = pyg.mouse.get_pos()
    rmx, rmy = scale_down(scale_h, wid, mx), scale_down(scale_v, high, my)
    print 'Mouse Pos:', (mx, my)
    print 'Mouse Color:', (int(abs(formR(rmx, rmy)) % 256),
           int(abs(formG(rmx, rmy)) % 256),
           int(abs(formB(rmx, rmy)) % 256))
    i = 0
    for x in xrange(wid):
        for y in xrange(high):
            i += 1
            percent = round((i / float(wid*high))*100.0)
            if percent != lastpercent:
                lastpercent = percent
                print str(int(percent))+'% Completed'
            rx, ry = scale_down(scale_h, wid, x), scale_down(scale_v, high, y)
            for func in [formR, formG, formB]:
                try: new[y][x] += [int(abs(func(rx, ry)) % 256)]
                except: new[y][x] += [0]
    print 'Finished.\n'
    return new

gridbag = upgrid(WIDTH, HEIGHT)

while True:
    mpos = pyg.mouse.get_pos()
    scale_range = [scale_h[1]-scale_h[0], scale_v[1]-scale_v[0]]
    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()
        if ev.type == 2:
            if ev.key == 27:
                pyg.quit()
                sys.exit()
            elif ev.key == 274:
                scale_v = [s+(scale_range[1]*translate) for s in scale_v]
            elif ev.key == 273:
                scale_v = [s-(scale_range[1]*translate) for s in scale_v]
            elif ev.key == 276:
                scale_h = [s-(scale_range[0]*translate) for s in scale_h]
            elif ev.key == 275:
                scale_h = [s+(scale_range[0]*translate) for s in scale_h]
            if ev.key in range(273, 277): gridbag = upgrid(WIDTH, HEIGHT)    
        if ev.type == 6:
            pressing[ev.button] = False
            gridbag = upgrid(WIDTH, HEIGHT)
        if ev.type == 5:
            pressing[ev.button] = True
            adds = [scale_down(s, c, 0) for s, c, v in zip([scale_h, scale_v], [WIDTH, HEIGHT], mpos)]
            adds = anchor
            if ev.button == 5:
                scale_h = [a * scale_zoom + adds[0] for a in scale_h]
                scale_v = [a * scale_zoom + adds[1] for a in scale_v]
            if ev.button == 4:
                scale_h = [a / scale_zoom + adds[0] for a in scale_h]
                scale_v = [a / scale_zoom + adds[1] for a in scale_v]
            print 'Grid Bounds:', scale_h, scale_v

    if pressing[1]:
        delta = [b - a for a, b in zip(mpos, lastStroke)]
        dscale = [scale_down(s, c, v) - s[0] for s, c, v in zip([scale_h, scale_v], [WIDTH, HEIGHT], delta)]
        scale_h = [v + dscale[0] for v in scale_h]
        scale_v = [v + dscale[1] for v in scale_v]
    lastStroke = mpos
    scrn.fill((255,255,255))

    if not pressing[1]:
        for x in xrange(WIDTH):
            for y in xrange(HEIGHT):
                scrn.set_at((x, y), gridbag[y][x])
    """scrn.set_at((scale_down(scale_v, HEIGHT, anchor[0]),
                 scale_down(scale_h, WIDTH, anchor[1])),
                (255, 255, 255))"""
    
    y = scale_up(scale_h, WIDTH, 0)
    x = scale_up(scale_v, HEIGHT, 0)
    pyg.draw.line(scrn, (0, 0, 200), (0, x), (WIDTH, x))
    pyg.draw.line(scrn, (0, 0, 200), (y, 0), (y, HEIGHT))

    pyg.display.update()
