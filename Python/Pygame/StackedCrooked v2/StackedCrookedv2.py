import pygame as pyg
import sys

class Mover:
    moves = []
    def __init__(this, moves=20, width=100, height=100, color=(0, 0, 0)):
        this.color = color
        this.width = abs(width)
        this.height = abs(height)
        this.moves = [[0, 0] for i in xrange(moves)]
        
    def realpos(this):
        return [int(sum([p[x] for p in this.moves])/float(len(this.moves))) for x in [0, 1]]
        
    def ison(this, pos):
        x, y = pos
        tx, ty = this.realpos()
        if x < tx: return False
        if y < ty: return False
        if x > tx + this.width: return False
        if y > ty + this.height: return False
        return True

    def moveto(this, pos):
        this.moves += [pos]
        del this.moves[0]

    def render(this, screen):
        pos = this.realpos()
        pyg.draw.rect(screen, this.color, pos+[this.width, this.height])

class OtherMover(Mover):
    cur = [0, 0]
    def __init__(this, speed=50, width=100, height=100, color=(0, 0, 0)):
        this.color = color
        this.width = abs(width)
        this.height = abs(height)
        this.speed = abs(speed)

    def realpos(this):
        return this.cur

    def moveto(this, pos):
        x, y = pos
        this.cur[0] += (x-this.cur[0])/float(this.speed)
        this.cur[1] += (y-this.cur[1])/float(this.speed)
                
pyg.init()

WIDTH, HEIGHT = 1000, 800
scrn = pyg.display.set_mode((WIDTH, HEIGHT))

squares = [
    Mover(100),
    OtherMover(100)
    ]

while True:
    mpos = pyg.mouse.get_pos()
    center = [p-50 for p in mpos]
    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()
            
    scrn.fill((255, 255, 255))
    for square in squares:
        square.moveto(center)
        square.render(scrn)
    pyg.display.update()
