import pygame as pyg
import sys
import math

pyg.init()

pressing = {i : False for i in xrange(512)}
clock = 0
WIDTH, HEIGHT = 800, 800
scrn = pyg.display.set_mode((WIDTH, HEIGHT))
errmsg = "Error!"
errlen = int((WIDTH / len(errmsg)) / 0.6)

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
            if ev.type == 3: pressing[ev.key] = False
            if ev.type == 5: pressing[ev.button] = True
            if ev.type == 6: pressing[ev.button] = False
            
        scrn.fill((255, 255, 255))
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
