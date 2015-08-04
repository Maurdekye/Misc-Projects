import pygame as pyg
import sys

class Clickable:
    def __init__(this, cor1, cor2, color, hover_color, click_color, use_func=None):
        this.cor1 = cor1
        this.cor2 = cor2
        this.color = color
        this.hover_color = hover_color
        this.click_color = click_color
        this.use_func = use_func
        
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
    
    def activate(this):
        if type(this.use_func) != type(lambda:0): return
        this.use_func()
        

    def render(this, scrn, clicks, offclicked):
        if this.isOver(pyg.mouse.get_pos()):
            if clicks[1]: pyg.draw.rect(scrn, this.click_color, this.coagulate())
            else: pyg.draw.rect(scrn, this.hover_color, this.coagulate())
            if offclicked: this.activate()
        else: pyg.draw.rect(scrn, this.color, this.coagulate())
    
class Button(Clickable):
    def __init__(this, cor1, cor2, color, hover_color, click_color, text='', text_color=(0, 0, 0), maxfontsize=26, use_func=None, use_two=None):
        this.cor1 = cor1 # Int Tuple Length 2
        this.cor2 = cor2 # Int Tuple Length 2
        this.color = color # Int Tuple Length 3
        this.hover_color = hover_color # Int Tuple Length 3
        this.click_color = click_color # Int Tuple Length 3
        this.text = text # String
        this.text_color = text_color # Int Tuple Length 3
        this.fontsize = maxfontsize # Int
        this.use_func = use_func # Function
        this.use_two = use_two # Function
        this.text_buffer = int(this.coagulate()[2]/10)

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

    def render(this, scrn, clicks, offclicked):
        if this.isOver(pyg.mouse.get_pos()):
            if clicks[1]: pyg.draw.rect(scrn, this.click_color, this.coagulate())
            else: pyg.draw.rect(scrn, this.hover_color, this.coagulate())
            if offclicked: this.activate()
        else: pyg.draw.rect(scrn, this.color, this.coagulate())
        scrn.blit(this.text_surf(), this.text_coag())

class Textbox(Clickable):

    text = ''
    selected = False
    surround_color = (170, 170, 245)
    select_color = (240, 240, 255)

    def select(this):
        for tbox in this.parent:
            tbox.selected = False
        this.selected = True
    
    def __init__(this, parent, cor1, cor2, fontsize=30, maxchars=10):
        this.parent = parent
        this.cor1 = cor1
        this.cor2 = cor2
        this.color = (240, 240, 240)
        this.hover_color = (230, 230, 230)
        this.click_color = (255, 255, 255)
        this.use_func = this.select
        this.parent += [this]

    def render(this, scrn, clicks, offclicked, keys):
        if this.isOver(pyg.mouse.get_pos()):
            if clicks[1]: pyg.draw.rect(scrn, this.click_color, this.coagulate())
            else: pyg.draw.rect(scrn, this.hover_color, this.coagulate())
            if offclicked: this.activate()
        else: pyg.draw.rect(scrn, this.color, this.coagulate())
        

class TimedText:
    def __init__(this, delay, pos, text='', size=26, color=(0,0,0)):
        this.delay = delay
        this.time = 0
        this.text = text
        this.size = size
        this.color = color
        this.pos = pos

    def advance(this):
        this.time -= 1
        if this.time <= 0: this.text = ''

    def setto(this, newtext):
        this.time = this.delay
        this.text = newtext

    def getsurf(this):
        return pyg.font.SysFont("monospace", this.size).render(this.text, 1, this.color)
        
    def render(this, scrn):
        scrn.blit(this.getsurf(), this.pos)
           
class PermaText:
    def __init__(this, pos, text='', size=26, color=(0,0,0)):
        this.text = text
        this.size = size
        this.color = color
        this.pos = pos
    
    def getsurf(this):
        return pyg.font.SysFont("monospace", this.size).render(this.text, 1, this.color)
        
    def render(this, scrn):
        scrn.blit(this.getsurf(), this.pos)

if __name__ == '__main__':
    pyg.init()
    pressing = {x : False for x in range(20)}
    scrn = pyg.display.set_mode((800, 600), 0, 32)
    
    toptext = TimedText(150, (200, 50))

    def redbutton(): toptext.setto("Red button clicked.")
    def greenbutton(): toptext.setto("You clicked the green button.")
    def bluebutton(): toptext.setto("Blue button was clicked.")
    def purpbutton(): toptext.setto("That was the purple button.")

    clickables = [
        Button((200, 100), (600, 150), (255, 0, 0), (220, 0, 0), (160, 80, 80), 'giant rhino dinos', use_func=redbutton),
        Button((200, 200), (600, 250), (0, 255, 0), (0, 220, 0), (80, 160, 80), 'senile flamboyant cedar trees', use_func=greenbutton),
        Button((200, 300), (600, 350), (0, 0, 255), (0, 0, 220), (80, 80, 160), 'arduous finnish cowboy calipers', text_color=(200, 200, 240), use_func=bluebutton),
        Button((200, 400), (600, 450), (255, 0, 255), (220, 0, 220), (160, 80, 160), 'dastardly duodrenous capscisity fraternalaltites', use_func=purpbutton)
        ]

    while True:
        toptext.advance()
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
        
        scrn.fill((255, 255, 255))
        for b in clickables:
            b.render(scrn, pressing, declicked)
        scrn.blit(toptext.getsurf(), (200, 50))
        pyg.display.update()
