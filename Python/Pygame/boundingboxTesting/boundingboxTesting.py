import pygame as pyg
import sys

pyg.init()

top = False
pressing = {x : False for x in xrange(1, 6)}
scrn = pyg.display.set_mode((640, 480), 0, 32)

class BoundBox(object):
    def __init__(this, x1, y1, x2, y2):
        this.x1 = x1
        this.y1 = y1
        this.x2 = x2
        this.y2 = y2

    def overlaps(this, box):
        if max(this.x2, this.x1) < min(box.x2, box.x1): return False
        if min(this.x2, this.x1) > max(box.x2, box.x1): return False
        if max(this.y2, this.y1) < min(box.y2, box.y1): return False
        if min(this.y2, this.y1) > max(box.y2, box.y1): return False
        return True

    def renderActual(this):
        return [this.x1, this.y1, this.x2 - this.x1, this.y2 - this.y1]
        
box1 = BoundBox(100, 100, 200, 200)
box2 = BoundBox(180, 180, 280, 280)

while True:
    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()
        elif ev.type == 5:
            pressing[ev.button] = True
            if ev.button == 1:
                top = False
                box1.x1, box1.y1 = pyg.mouse.get_pos()
            elif ev.button == 3:
                top = True
                box2.x1, box2.y1 = pyg.mouse.get_pos()
        elif ev.type == 6:
            pressing[ev.button] = False
            print box1.x1, box1.x2, "->", box2.x2 # REMOVE ME

    if pressing[1]:
        box1.x2, box1.y2 = pyg.mouse.get_pos()
    elif pressing[3]:
        box2.x2, box2.y2 = pyg.mouse.get_pos()
    
    scrn.fill((255, 255, 255))
    if top: pyg.draw.rect(scrn, (255, 0, 0), box1.renderActual())
    pyg.draw.rect(scrn, (0, 0, 255), box2.renderActual())
    if not top: pyg.draw.rect(scrn, (255, 0, 0), box1.renderActual())
    scrn.blit(pyg.font.SysFont("monospace", 25).render(str(box1.overlaps(box2)), 1, (0,0,0)), (0,0))
    
    pyg.display.update()
