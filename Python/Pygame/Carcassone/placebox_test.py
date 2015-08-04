import pygame as pyg
import sys
import color

pyg.init()

WIDTH, HEIGHT = 800, 800
grid = 10, 10
scrn = pyg.display.set_mode((WIDTH, HEIGHT))

squares = [[0, 0]]

while True:

    griddim = [int(d / g) for d, g in zip((WIDTH, HEIGHT), grid)]
    gridpos = [int(p - p%d) for p, g, d in zip(pyg.mouse.get_pos(), grid, griddim)]
    gridmouse = [int(p/d) for p, d in zip(pyg.mouse.get_pos(), griddim)]

    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()
        elif ev.type == 5:
            squares.append(gridmouse)
            print gridmouse

    scrn.fill(color.white)
    for p in squares:
        p = [c*d for c, d in zip(p, griddim)]
        pyg.draw.rect(scrn, color.blue, p + griddim)
    pyg.draw.rect(scrn, color.yellow, gridpos + griddim)
    pyg.display.update()
