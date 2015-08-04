import pygame as pyg
import sys

pyg.init()
pressing = {x : False for x in xrange(512)}
scrn = pyg.display.set_mode((600, 800), 0, 32)

class Button:
    def __init__(this, cor1, cor2, color, hover_color=None, click_color=None, text='', text_color=(0, 0, 0), maxfontsize=26, use_func=None):
        this.cor1 = cor1 # Int Tuple Length 2
        this.cor2 = cor2 # Int Tuple Length 2
        this.color = color # Int Tuple Length 3
        this.hover_color = hover_color # Int Tuple Length 3
        if hover_color == None: this.hover_color = color
        this.click_color = click_color # Int Tuple Length 3
        if click_color == None: this.click_color = color
        this.text = text # String
        this.text_color = text_color # Int Tuple Length 3
        this.fontsize = maxfontsize # Int
        this.use_func = use_func # Function
        this.text_buffer = int(this.coagulate()[2]/10)

    def isOver(this, pos):
        if pos[0] > max(this.cor1[0], this.cor2[0]): return False
        if pos[0] < min(this.cor1[0], this.cor2[0]): return False
        if pos[1] > max(this.cor1[1], this.cor2[1]): return False
        if pos[1] < min(this.cor1[1], this.cor2[1]): return False
        return True

    def coagulate(this):
        return [min(this.cor1[0], this.cor2[0]),
                min(this.cor1[1], this.cor2[1]),
                max(this.cor1[0], this.cor2[0]) - min(this.cor1[0], this.cor2[0]),
                max(this.cor1[1], this.cor2[1]) - min(this.cor1[1], this.cor2[1])]

    def text_surf(this):
        if this.fontsize <= 0: raise Exception('Font size too small!')
        surf = pyg.font.SysFont("monospace", this.fontsize).render(this.text, 1, this.text_color)
        while surf.get_width() > this.coagulate()[2] - this.text_buffer:
            this.fontsize -= 1
            if this.fontsize <= 0: raise Exception('Font size too small!')
            surf = pyg.font.SysFont("monospace", this.fontsize).render(this.text, 1, this.text_color)
        return surf

    def text_coag(this):
        txt = this.text_surf()
        butn = this.coagulate()
        return ( butn[0] + ((butn[2]/2) - (txt.get_width()/2)), butn[1] + ((butn[3]/2) - (txt.get_height()/2)) )

    def activate(this):
        if type(this.use_func) != type(lambda:0): return
        this.use_func()

    def render(this, scrn, clicks, offclicked):
        if this.isOver(pyg.mouse.get_pos()):
            if clicks[1]: pyg.draw.rect(scrn, this.click_color, this.coagulate())
            else: pyg.draw.rect(scrn, this.hover_color, this.coagulate())
            if offclicked: this.activate()
        else: pyg.draw.rect(scrn, this.color, this.coagulate())
        scrn.blit(this.text_surf(), this.text_coag())

class Movable:
    def __init__(this, cur, aim, speed=10, defbuffer=0.01):
        this.cur = float(cur) # Float
        this.aim = float(aim) # Float
        this.speed = float(speed) # Int
        this.defbuffer = defbuffer # Float < 1

    def advance(this):
        this.aim = float(this.aim)
        if abs(this.cur - this.aim) < this.defbuffer:
            this.cur = this.aim
        this.cur += (this.aim - this.cur) / this.speed

class StackPiece:
    def __init__(this, parentStack, attr):
        this.attr = attr
        this.parentStack = parentStack
        this.parentStack.stack += [this]
        ind = this.parentStack.stack.index(this)
        this.move = Movable(ind, ind)

    def advance(this):
        this.move.aim = this.parentStack.stack.index(this)
        this.move.advance()
    
class Crooked_Stack:
    stack = []

    def add(this, attr):
        StackPiece(this, attr)
        this.sortStack()
    
    def sortStack(this):
        newstack = []
        for piece in this.stack:
            for i in xrange(len(newstack)):
                if newstack[i].attr > piece.attr:
                    newstack.insert(i, piece)
                    break
            else: newstack += [piece]
        this.stack = newstack

    def advance(this):
        for piece in this.stack:
            piece.advance()
        print len(this.stack)

    def render(this, scrn):
        colorattrs = {
            1 : (0, 0, 255),
            2 : (255, 240, 0),
            3 : (255, 0, 0)
            }
        for i in range(len(this.stack))[::-1]:
            pyg.draw.rect(scrn, colorattrs[this.stack[i].attr],
                          (100 * int(this.stack[i].move.cur // 16), int((this.stack[i].move.cur % 16.0) * 50), 100, 50))


crookstack = Crooked_Stack()

def addRed(): crookstack.add(3)
def addYel(): crookstack.add(2)
def addBlu(): crookstack.add(1)

clickables = [
    Button((400, 20), (580, 70), (255, 0, 0), (220, 0, 0), (255, 100, 100), 'Add Red to Stack', use_func=addRed),
    Button((400, 90), (580, 140), (255, 240, 0), (220, 207, 0), (255, 250, 100), 'Add Yellow to Stack', use_func=addYel),
    Button((400, 160), (580, 210), (0, 0, 255), (0, 0, 200), (100, 100, 255), 'Add Blue to Stack', use_func=addBlu),
    ]

while True:
    declicked = False
    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()
        if ev.type == 5:
            pressing[ev.button] = True
        if ev.type == 6:
            pressing[ev.button] = False
            declicked = True
        if ev.type == 2:
            if len(crookstack.stack) != 0:
                if ev.key == 8:
                    del crookstack.stack[-1]
                if ev.key == 127:
                    del crookstack.stack[0]
    scrn.fill((255, 255, 255))
    for b in clickables:
        b.render(scrn, pressing, declicked)
    crookstack.advance()
    crookstack.render(scrn)
    pyg.display.update()
