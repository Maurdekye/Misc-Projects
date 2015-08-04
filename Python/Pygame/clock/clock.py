import pygame as pyg
import sys
import time
import math

def drawtext(screen, text, pos, size=25):
    screen.blit(pyg.font.SysFont("monospace", size).render(str(text), 0, (0,0,0)), pos)
    
pyg.init()

WIDTH = 1920
HEIGHT = 1080
scrn = pyg.display.set_mode((WIDTH, HEIGHT), pyg.FULLSCREEN)
pi = math.pi
midx = WIDTH / 2
midy = HEIGHT / 2
mid = (min(WIDTH, HEIGHT))/2
    
while True:
    second = time.gmtime().tm_sec/60.0
    minute = time.gmtime().tm_min/60.0 + (second/60.0)
    hour = ((time.gmtime().tm_hour+2)%12)/12.0 + (minute/12.0)
    clock = time.clock() % pi/12
    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()
        if ev.type == 2:
            if ev.key == 27:
                pyg.quit()
                sys.exit()
    
    scrn.fill((255, 255, 255))
    # Outline
    pyg.draw.circle(scrn, (0, 0, 0), [midx, midy], mid)
    pyg.draw.circle(scrn, (255,255,255), [midx, midy], mid-4)
    # Numbers
    for i in xrange(1, 13):
        drawtext(scrn, i, [math.sin((12-i)/(1.9) + pi)*(mid*0.9) + (midx*0.95),
                           math.cos((12-i)/(1.9) + pi)*(mid*0.9) + (midy*0.95)], mid/8)
    # Hands
    # - Second
    pyg.draw.line(scrn, (255, 0, 0), [midx, midy],
                  [math.sin((1-second*2)*pi)*(mid*0.8) + midx,
                   math.cos((1-second*2)*pi)*(mid*0.8) + midy], 2)
    # - Minute
    pyg.draw.line(scrn, (0, 0, 0), [midx, midy],
                  [math.sin((1-minute*2)*pi)*(mid*0.8) + midx,
                   math.cos((1-minute*2)*pi)*(mid*0.8) + midy], 4)
    # - Hour
    pyg.draw.line(scrn, (0, 0, 0), [midx, midy],
                  [math.sin((1 - hour)*pi*2)*(mid*0.7) + midx,
                   math.cos((1 - hour)*pi*2)*(mid*0.7) + midy], 6)
    pyg.display.update()
