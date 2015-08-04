import pygame as pyg
import sys
from random import randint as rnd
import math


pyg.init()

scrn = pyg.display.set_mode((800, 600), 0, 32)
bg = pyg.image.load("bg.jpg").convert()

def dist(point1, point2):
    const1 = point2[0] - point1[0]
    const2 = point2[1] - point1[1]
    return (const1**2 + const2**2)**0.5


def weighted(points):
    to_sender = {x : 0 for x in points}
    for p in points:
        for cp in points:
            if p != cp:
                to_sender[p] += 1/dist(p, cp)
    return to_sender

points = [(rnd(1,800), rnd(1,600)) for x in range(200)]
weights = weighted(points)
for p in weights:
    print(p, weights[p])

while True:
    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()
    scrn.blit(bg, (0, 0))
    for p in weights:
        pyg.draw.circle(scrn, (255, 0, 0), p, int(weights[p] * 10))
    pyg.display.update()
