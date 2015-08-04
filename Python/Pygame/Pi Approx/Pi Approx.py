import pygame as pyg
import sys
import math

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
pixels = 10
radius = 400

while True:
    pixel_size = SIZE / float(pixels)
    mouse_pixel = [int(mp - mp%pixel_size) / pixel_size for mp in pyg.mouse.get_pos()]
    center = pixels / 2
    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()
        if ev.type == 2:
            change = 1
            if ev.key == 273:
                pixels = min(SIZE, pixels + 1)
            elif ev.key == 274:
                pixels = max(1, pixels - 1)
            else:
                change = 0
            if change: print pixels
            

    scrn.fill((255, 255, 255))
    center = SIZE / 2
    filled = 0
    for x in xrange(pixels):
        for y in xrange(pixels):
            cx, cy = [p*pixel_size + pixel_size/2 for p in [x, y]]
            if distance(cx, cy, center, center) < radius:
                draw_pixel(scrn, pixel_size, (x, y), (255, 0, 0))
                filled += 1
    ratio = float(filled)/(pixels**2)
    drawtext(scrn, str(pixels), (0, SIZE-22))
    drawtext(scrn, str(ratio), (0, 0))
    pyg.display.update()
