import pygame as pyg
import sys

pyg.init()

WIDTH, HEIGHT = 800, 600
scrn = pyg.display.set_mode((WIDTH, HEIGHT))
bg = pyg.image.load("bg.jpg").convert()

bar1 = 2
bar2 = 2

while True:
    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()
        if ev.type == 5:
            if ev.button == 1: 
                bar1 += 20.0
            if ev.button == 3: 
                bar2 += 20.0
    
    bar1 -= bar1 / 2000.0
    bar2 -= (HEIGHT - bar2) / 10000.0 + 0.00001
    bar2 = max(bar2, 0)
    
    b1real = max(min(HEIGHT, bar1), 0)
    b2real = max(min(HEIGHT, bar2), 0)
    
    scrn.blit(bg, (0, 0))
    pyg.draw.rect(scrn, (255, 0, 0), [100, 0, 100, b1real])
    pyg.draw.rect(scrn, (0, 0, 255), [400, 0, 100, b2real])
    pyg.display.update()
