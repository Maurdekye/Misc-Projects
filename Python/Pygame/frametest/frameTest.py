import pygame as pyg
import sys

class Anim(object):
    def __init__(this, frames, frame_ord):
        this.frames = frames # Pygame Surface List
        this.frame_ord = frame_ord # Int : Int Dictionary
        this.curframe = 0

    def get_frame(this):
        return this.frames[this.curframe]

    def next_frame(this):
        this.curframe = this.frame_ord[this.curframe]

pyg.init()

scrn = pyg.display.set_mode((800, 600), 0, 32)
bg = pyg.image.load("bg.jpg").convert()

stickman = Anim([
    pyg.image.load("stickman/idle.png").convert_alpha(),
    pyg.image.load("stickman/right_1.png").convert_alpha(),
    pyg.image.load("stickman/right_2.png").convert_alpha(),
    pyg.image.load("stickman/right_3.png").convert_alpha(),
    pyg.image.load("stickman/left_1.png").convert_alpha(),
    pyg.image.load("stickman/left_2.png").convert_alpha(),
    pyg.image.load("stickman/left_3.png").convert_alpha(),
    pyg.image.load("stickman/jump_1.png").convert_alpha(),
    pyg.image.load("stickman/jump_2.png").convert_alpha(),
    pyg.image.load("stickman/jump_3.png").convert_alpha()
    ], {
        0 : 0,
        1 : 2,
        2 : 3,
        3 : 1,
        4 : 5,
        5 : 6,
        6 : 4,
        7 : 8,
        8 : 9,
        9 : 0
        }
    )

ticks = 0
while True:
    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()
        if ev.type == 2:
            ticks = 0
            if ev.key == 275:
                stickman.curframe = 1
            if ev.key == 276:
                stickman.curframe = 4
            if ev.key == 273:
                stickman.curframe = 7
        if ev.type == 3:
            stickman.curframe = 0

    ticks += 1
    if ticks >= 100:
        ticks = 0
        stickman.next_frame()
    scrn.blit(bg, (0, 0))
    scrn.blit(stickman.get_frame(), (200, 200))
    pyg.display.update()
