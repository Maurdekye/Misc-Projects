import pygame as pyg
import sys
import math

class Square:
    '''
    Types
        0 - Grass
        1 - Road
        2 - City
    
    Attributes
        0 - Tenant
        1 - Chapel
    '''
    def __init__(this, colors, sims, tags):
        this.colors = colors # Int List, 4 Long
        this.sims  = sims # Int List, 4 Long
        this.tags = tags # Int Set

def draw_pixel(screen, pixel_size, pos, color=(0, 0, 0)):
    pos = [p*pixel_size for p in pos]
    pyg.draw.rect(screen, color, pos + [pixel_size + 1, pixel_size + 1])

def distance(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def drawtext(screen, text, pos):
    screen.blit(pyg.font.SysFont("monospace", 25).render(str(text), 0, (0,0,0)), pos)

pyg.init()

SIZE = 800
scrn = pyg.display.set_mode((SIZE, SIZE))
pixels = 3

placed = []

while True:
    pixel_size = SIZE / float(pixels)
    mouse_pixel = [int(round(int(mp - mp%pixel_size) / pixel_size)) for mp in pyg.mouse.get_pos()]
    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()
        if ev.type == 5:
            if ev.button == 1:
                placed += [mouse_pixel]
                print pixels, mouse_pixel
                # --- Size Shifts
                if 0 in mouse_pixel:
                    placed = [[a+1, b+1] for a, b in placed]
                    pixels += 1
                if pixels-1 in mouse_pixel:
                    pixels += 1

    scrn.fill((255, 255, 255))
    for p in placed:
        draw_pixel(scrn, pixel_size, p, (0, 0, 255))
    draw_pixel(scrn, pixel_size, mouse_pixel, (255, 0, 0))
    pyg.display.update()
