import pygame as pyg
import sys

class Pixelset:
    pixels = []

    def is_in(this, pos, color=None):
        for p, c in this.pixels:
            if p == pos:
                if color == None:
                    return True
                else:
                    if color == c:
                        return True
                    else:
                        return False
        return False

    def add(this, pos, color=(0, 0, 0), overwrite=True):
        if overwrite or not this.is_in(pos):
            this.pixels += [[pos, color]]

    def remove(this, pos, color=None):
        for i, (p, c) in enumerate(this.pixels):
            if p == pos:
                if color == None:
                    del this.pixels[i]
                else:
                    if c == color:
                        del this.pixels[i]
                    else:
                        break

    def render(this, screen, getdraw, adj):
        for p, c in this.pixels:
            pyg.draw.rect(screen, c, getdraw(p) + adj)

    def clear(this):
        this.pixels = []

def kill():
    pyg.quit()
    sys.exit()

pressing = {x : False for x in range(10)}

pyg.init()

WIDTH, HEIGHT = 512, 512
screen = pyg.display.set_mode((WIDTH, HEIGHT))

hori_pixels = 32
vert_pixels = 32
WH = [WIDTH, HEIGHT]
hpvp = lambda: [hori_pixels, vert_pixels]
adj = lambda: [d / float(p) for d, p in zip(WH, hpvp())]
getpix = lambda pos: [int(p / pix) for p, pix in zip(pos, adj())]
getdraw = lambda pos: [int(p * pix) for p, pix in zip(pos, adj())]

pixels = Pixelset()

while True:
    
    mpos = pyg.mouse.get_pos()
    draw_mpos = [pos - (pos % pix) for pos, pix, dim in zip(mpos, adj(), WH)]
    pix_mpos = getpix(draw_mpos)
    
    for e in pyg.event.get():
        if e.type == 12:
            kill()
        if e.type == 2:
            print(e.key)
            if e.key == 27:
                kill()
            if e.key == 99:
                pixels.clear()
        if e.type == 5:
            pressing[e.button] = True
        if e.type == 6:
            pressing[e.button] = False

    screen.fill((255, 255, 255))
    
    if pressing[1]:
        pixels.add(pix_mpos, (0, 0, 0))
    if pressing[2]:
        pixels.add(pix_mpos, (255, 0, 0))
    if pressing[3]:
        pixels.remove(pix_mpos)

    pixels.render(screen, getdraw, adj())

    pyg.display.update()
