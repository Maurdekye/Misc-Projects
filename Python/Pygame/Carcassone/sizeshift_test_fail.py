import pygame as pyg
import sys
import color

pyg.init()

WIDTH, HEIGHT = 800, 800
DIM = WIDTH, HEIGHT
grid = [[0, 0], [0, 0]]
scrn = pyg.display.set_mode(DIM)

squares = [[0, 0]]

while True:

    for x, y in squares:
        grid[0][0] = min(grid[0][0], x-1)
        grid[0][1] = min(grid[0][1], y-1)
        grid[1][0] = max(grid[1][0], x+1)
        grid[1][1] = max(grid[1][1], y+1)
    
    print grid

    gridrange = [grid[1][0] - grid[0][0], grid[1][1] - grid[0][1]]
    griddim = [int(d / r) for d, r in zip(DIM, gridrange)]
    gridpos = [int(p - p%d) + r for p, d, r in zip(pyg.mouse.get_pos(), griddim, gridrange)]
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
