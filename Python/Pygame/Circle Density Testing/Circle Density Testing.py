import pygame as pyg
import sys
from math import pi, sin, cos

def tau(): return 2 * pi

def drawtext(text, pos):
    scrn.blit(pyg.font.SysFont("monospace", 25).render(str(text), 0, (0,0,0)), pos)

pyg.init()

scrn = pyg.display.set_mode((800, 600), 0, 32)
bg = pyg.image.load("bg.jpg").convert()

density = 3.0

while True:
    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()
        if ev.type == 2:
            if ev.key == 274: density -= 1.0
            if ev.key == 273: density += 1.0
            density = min(max(density, 3.0), 100.0)
    scrn.blit(bg, (0, 0))
    for i in range(int(density)):
        varA = (i / density) * tau()
        varB = ((i + 1) / density) * tau()
        pyg.draw.line(scrn, (0, 0, 0),
                      ((sin(varA) * 100) + 200, (cos(varA) * 100) + 200),
                      ((sin(varB) * 100) + 200, (cos(varB) * 100) + 200))
        pyg.draw.line(scrn, (0, 0, 0),
                      ((sin(varA) * 150) + 200, (cos(varA) * 150) + 200),
                      ((sin(varB) * 150) + 200, (cos(varB) * 150) + 200))
        pyg.draw.line(scrn, (0, 0, 0),
                      ((sin(varA) * 100) + 200, (cos(varA) * 100) + 200),
                      ((sin(varA) * 150) + 200, (cos(varA) * 150) + 200))
        
    drawtext(int(density), (190, 190))
    pyg.display.update()
