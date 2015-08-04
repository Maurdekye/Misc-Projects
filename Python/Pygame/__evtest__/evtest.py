import pygame as pyg
import sys

pyg.init()

scrn = pyg.display.set_mode((600, 600), 0, 32)
bg = pyg.image.load("bg.jpg").convert()

while True:
    acts = ''
    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()
        else: acts += str(ev.type)
        if ev.type in [2, 3]: acts += ':' + str(ev.key)
        if ev.type in [5, 6]: acts += ':' + str(ev.button)
        acts += ' '
    if len(acts) != 0: print(acts)

    scrn.blit(bg, (0, 0))
    pyg.display.update()

'''
Event Types
1 - Mouse enter / leave window
2 - Key Press
3 - Key Release
4 - Mouse Movement
5 - Mouse Click
6 - Mouse Release
12 - X button
16 - Pygame Initialize
17 - Window in focus

Button References
1 - Left Click
2 - Middle Click
3 - Right Click
4 - Scroll Up
5 - Scroll Down

Key References
9 - Tab
27 - Esc
48-57 - 0-9
97 - A
100 - D
115 - S
119 - W
273 - Up
274 - Down
275 - Left
276 - Right
303 - Right Shift
304 - Left Shift
305 - Right Ctrl
306 - Left Ctrl
'''
