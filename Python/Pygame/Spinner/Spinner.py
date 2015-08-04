import pygame as pyg
import sys
import math
import random

def drawpointer(scrn, pos, length, angle):
    angle = angle/(180/math.pi)
    modifpos = (pos[0] + math.sin(angle)*length,
                  pos[1] + math.cos(angle)*length)
    pyg.draw.line(scrn, (200, 100, 150), pos, modifpos, 6)

pyg.init()

pressing = {i : False for i in range(512)}
clock = 0
WIDTH, HEIGHT = 800, 800
scrn = pyg.display.set_mode((WIDTH, HEIGHT))
spinner = pyg.image.load("spinner_picture.png").convert()
errmsg = "Error!"
errlen = int((WIDTH / len(errmsg)) / 0.6)

momentum = 0
curang = 0

try:
    while True:
        clock += 1
        for ev in pyg.event.get():
            if ev.type == 12:
                pyg.quit()
                sys.exit()
            if ev.type == 2:
                if ev.key == 27:
                    pyg.quit()
                    sys.exit()
                pressing[ev.key] = True
                if ev.key == 273:
                    momentum += random.random() * 7 + 3
            if ev.type == 3: pressing[ev.key] = False
            if ev.type == 5: pressing[ev.button] = True
            if ev.type == 6: pressing[ev.button] = False

        curang += momentum
        momentum = max(momentum - 0.01, 0)
            
        scrn.fill((255, 255, 255))
        scrn.blit(spinner, (0, 0))
        drawpointer(scrn, (395, 392), 250, curang)
        scrn.blit(pyg.font.SysFont("monospace", 25).render(str(curang), 1, (0, 0, 0)), (0, 0))
        pyg.display.update()
except Exception as ex:
    import traceback
    traceback.print_exc()
    while True:
        for ev in pyg.event.get():
            if ev.type == 12:
                pyg.quit()
                sys.exit()
        clock += 1
        scrn.blit(pyg.font.SysFont("monospace", errlen).render(errmsg, 1, (math.sin(clock/40.0)*100 + 154, 0, 0)), (10, HEIGHT/2 - (errlen/2)))
        pyg.display.update()

        
            
