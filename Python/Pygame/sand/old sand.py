import pygame as pyg
import sys
import random
import copy
import math

class Particle:
    updating = True
    def __init__(this, pos, color):
        this.pos = pos
        this.color = color

class Part_Sys:
    particles = []
    def __init__(this, wid, hih):
        this.wid = wid
        this.hih = hih
        
    def add(this, pos):
        this.particles += [[a for a in pos]]
    
    def rem(this, pos):
        if pos in this.particles:
            this.particles[this.particles.index(pos)]

    def update(this):
        oldparticles = copy.deepcopy(this.particles)
        this.particles = []
        for i, (x, y) in enumerate(oldparticles):
            if y < this.hih-1 and x > 0 and x < this.wid-1:
                if [x, y+1] not in oldparticles:
                    this.particles += [[x, y+1]]
                elif [x-1, y+1] not in oldparticles:
                    this.particles += [[x-1, y+1]]
                elif [x+1, y+1] not in oldparticles:
                    this.particles += [[x+1, y+1]]
                else:
                    this.particles += [[x, y]]
            else:
                this.particles += [[x, y]]
            
                
    def render(this, scrn, color=(0, 0, 0)):
        for x, y in this.particles:
           scrn.set_at((x, y), color)

pyg.init()

pressing = {x:False for x in xrange(500)}
WIDTH, HEIGHT = 200, 200
scrn = pyg.display.set_mode((WIDTH, HEIGHT))

colorcounter = 0
getcol = lambda: ( math.sin(colorcounter) + 1 ) / 2.0
psystem = Part_Sys(WIDTH, HEIGHT)
        
while True:
    mpos = pyg.mouse.get_pos()
    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()
        if ev.type == 2:
            pressing[ev.key] = True
        if ev.type == 3:
            pressing[ev.key] = False
        if ev.type == 5:
            pressing[ev.button] = True
        if ev.type == 6:
            pressing[ev.button] = False

    if pressing[1]:
        psystem.add(list(mpos))
        colorcounter += math.pi / 10000.0
        
    scrn.fill((255, 255, 255))
    psystem.render(scrn, (getcol()*100 + 155, getcol()*150 + 50, getcol()*150 + 50))
    psystem.update()
    pyg.display.update()
