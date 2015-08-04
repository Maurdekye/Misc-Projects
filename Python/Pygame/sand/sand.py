import pygame as pyg
import sys
import random
import copy
import math

class Particle:
    def __init__(this, pos, color):
        this.pos = [a for a in pos]
        this.color = color

class Part_Sys:
    particles = {}
    nextupdate = []
    curid = 0
    def __init__(this, wid, hih):
        this.wid = wid
        this.hih = hih
        
    def add(this, pos, color=(0, 0, 0)):
        this.particles[this.curid] = Particle(pos, color)
        this.nextupdate += [this.curid]
        this.curid += 1
    
    def rem(this, pos):
        for k, p in this.particles:
            if p.pos == pos:
                del this.particles[k]

    def update(this):
        finals = []
        partlist = [p.pos for p in this.particles.values()]
        for i in this.nextupdate:
            x, y = this.particles[i].pos
            if y < this.hih-1 and x > 0 and x < this.wid-1 :
                if [x, y+1] not in partlist:
                    finals += [i]
                    this.particles[i].pos = [x, y+1]
                elif [x-1, y+1] not in partlist:
                    finals += [i]
                    this.particles[i].pos = [x-1, y+1]
                elif [x+1, y+1] not in partlist:
                    finals += [i]
                    this.particles[i].pos = [x+1, y+1]
        this.nextupdate = copy.deepcopy(finals)
                
    def render(this, scrn):
        for p in this.particles.values():
           scrn.set_at(p.pos, p.color)

pyg.init()

pressing = {x:False for x in xrange(500)}
WIDTH, HEIGHT = 1920, 1080
scrn = pyg.display.set_mode((WIDTH, HEIGHT), pyg.FULLSCREEN)

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
        psystem.add(list(mpos), (getcol()*150 + 50, getcol()*150 + 50, getcol()*150 + 50))
        colorcounter += math.pi / 10000.0
    if pressing[99]:
        psystem.particles, psystem.nextupdate = {}, []
        
    scrn.fill((255, 255, 255))
    psystem.render(scrn)
    psystem.update()
    pyg.display.update()
