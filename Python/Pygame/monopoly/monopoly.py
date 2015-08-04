import pygame as pyg
import sys
from random import randint as rand
import math

class Proper(object):
    def __init__(this, rent, houserents, hotelrent, cost, housecosts, hotelcost, prop_set, set_max):
        this.rent = rent # Int
        this.houserents = houserents # Int Array, Length 4
        this.hotelrent = hotelrent # Int
        this.cost = cost # Int
        this.housecost = housecost # Int
        this.hotelcost = hotelcost # Int
        this.prop_set = prop_set # String
        this.set_max = set_max # Int
        this.houses = 0
        this.has_hotel = False
        this.is_morgatged = False

    def get_rent(this, ply):
        if not this.hasFullSet(ply):
            return this.rent
        if this.has_hotel:
            return this.hotelrent
        if this.houses > 0:
            return this.houserents[this.houses-1]
        if this.is_morgatged:
            return 0
        else:
            return this.rent*2

    def add_house(this, ply):
        if this.houses >= 4 or this.has_hotel:
            return 1
        if ply.money < this.housecost:
            return 2
        if not this.hasFullSet(ply):
            return 3
        else:
            ply.money -= this.housecost
            this.houses += 1
            return 0

    def add_hotel(this, ply):
        if this.has_hotel:
            return 1
        if this.houses < 4:
            return 2
        if ply.money < this.hotelcost:
            return 3
        else:
            ply.money -= this.hotelcost
            this.houses = 0
            this.has_hotel = True
            return 0

    def morgatge(this, ply):
        if this.is_morgatged:
            return 1
        else:
            this.is_morgatged = True
            ply.money += this.cost/2
            return 0

    def repay_morgatge(this, ply):
        mcost = this.get_mcost()
        if not this.is_morgatged:
            return 1
        if ply.money < mcost:
            return 2
        else:
            ply.money -= mcost
            this.is_morgatged = False
            return 0

    def get_mcost(this):
        return int((this.cost/2)*1.1)

    def hasFullSet(this, ply):
        count = 0
        for p in ply.properties:
            if p.set == this.set:
                count += 1
                if count >= this.set_max:
                    return True
        return False

class Railroad(Proper):
    def __init__(this, rent, houserents, hotelrent, cost, housecosts, hotelcost, prop_set, set_max):
        this.rent = rent # Int
        this.houserents = houserents # Int Array, Length 4
        this.hotelrent = hotelrent # Int
        this.cost = cost # Int
        this.housecost = housecost # Int
        this.hotelcost = hotelcost # Int
        this.prop_set = prop_set # String
        this.set_max = set_max # Int
        this.houses = 0
        this.has_hotel = False
        this.is_morgatged = False

    def get_rent(this, ply):
        rent = 25
        for p in ply.properties:
            if isinstance(p, Railroad):
                rent *= 2
        return rent

class Resource(Proper):
    def __init__(this, rent, houserents, hotelrent, cost, housecosts, hotelcost, prop_set, set_max):
        this.rent = rent # Int
        this.houserents = houserents # Int Array, Length 4
        this.hotelrent = hotelrent # Int
        this.cost = cost # Int
        this.housecost = housecost # Int
        this.hotelcost = hotelcost # Int
        this.prop_set = prop_set # String
        this.set_max = set_max # Int
        this.houses = 0
        this.has_hotel = False
        this.is_morgatged = False

    def get_rent(this, ply):
        for p in ply.properties:
            if isinstance(p, Resource): break
        else:
            return 10 * roll_one()
        return 4 * roll_one()

class Player(object):
    def __init__(this, name, img):
        this.name = name # String
        this.img = img # Pygame Surface
        this.boardpos = 0
        this.money = 1500
        this.properties = []
        this.hasGOOJFC = False
        this.in_jail = False

    def move(this, amount):
        def dist((x1, y1), (x2, y2)):
            return ( (x1 - x2)**2 + (y1 - y2)**2 )**0.5
        if this.in_jail:
            this.boardpos = 0
            return False
        passed_go = False
        for i in xrange(amount):
            cur = this.get_pixelpos()
            this.boardpos += 1
            end = this.get_pixelpos()
            this.boardpos -= 1
            while dist(end, cur) > 1:
                cur = (cur[0] + (end[0] - cur[0]) / 4.0, cur[1] + (end[1] - cur[1]) / 4.0)
                pyg.time.wait(10)
                blit_all(this
                         )
                scrn.blit(this.img, cur)
                pyg.display.update()
            this.boardpos += 1
        if this.boardpos >= 40:
            passed_go = True
            this.money += 200
        this.boardpos = this.boardpos % 40
        return passed_go

    def moveroll(this):
        rolls = roll_two()
        this.move(sum(rolls))
        doubles = 0
        while rolls[0] == rolls[1]:
            doubles += 1
            rolls = roll_two()
            
            if doubles >= 3:
                this.in_jail = True
                break
            else: this.move(sum(rolls))
    
    def get_tenpercent_totalwealth(this):
        cost = this.money
        for p in this.properties:
            cost += p.housecost * p.houses
            if p.has_hotel: cost += p.hotelcost
            cost += p.cost
        return int(cost * 0.1)
    
    def get_pixelpos(this):
        if this.in_jail:
            return (142, 804)
        pos = this.boardpos % 40
        if pos == 0:
            return (850, 800)
        if pos <= 9:
            return (800 - (pos * 64), 830)
        if pos == 10:
            return (102, 840)
        if pos <= 19:
            return (120, 800 - ((pos-10) * 66))
        if pos == 20:
            return (124, 124)
        if pos <= 29:
            return (((pos-18) * 66) + 20, 115)
        if pos == 30:
            return (825, 118)
        if pos <= 39:
            return (850, ((pos-28) * 66))

class Button(object):
    def __init__(this, cor1, cor2, color, hover_color, text, use_func=None):
        this.cor1 = cor1 # Int Tuple Length 2
        this.cor2 = cor2 # Int Tuple Length 2
        this.color = color # Int Tuple Length 3
        this.hover_color = hover_color # Int Tuple Length 3
        this.text = text # String
        this.use_func = use_func # Function

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
        return pyg.font.SysFont("monospace", 25).render(this.text, 1, (0,0,0))

    def text_coag(this):
        txt = this.text_surf()
        butn = this.coagulate()
        return ( butn[0] + ((butn[2]/2) - (txt.get_width()/2)), butn[1] + ((butn[3]/2) - (txt.get_height()/2)) )
                
pyg.init()
scrn = pyg.display.set_mode((1000, 1000), 0, 32)

cply = 0

players = [
    Player("Jakhob", pyg.image.load("peices/thimble.png").convert_alpha())
    ]

buttons = [
    Button((350, 350), (650, 300), (255,0,0), (255,255,0), "Roll", lambda num : players[num].moveroll())
    ]

dies = {
    1 : pyg.image.load("die/one.png").convert_alpha(),
    2 : pyg.image.load("die/two.png").convert_alpha(),
    3 : pyg.image.load("die/three.png").convert_alpha(),
    4 : pyg.image.load("die/four.png").convert_alpha(),
    5 : pyg.image.load("die/five.png").convert_alpha(),
    6 : pyg.image.load("die/six.png").convert_alpha()
    }

def blit_all(plyNoFrame=False):
    scrn.fill((255,255,255))
    scrn.blit(pyg.image.load("board.jpg").convert(), (100, 100))
    for ply in players:
        if isinstance(plyNoFrame, Player):
            if ply.name != plyNoFrame.name:
                scrn.blit(ply.img, ply.get_pixelpos())
        else: scrn.blit(ply.img, ply.get_pixelpos())

def render_buttons():
    for b in buttons:
        if b.isOver(pyg.mouse.get_pos()):
            pyg.draw.rect(scrn, b.hover_color, b.coagulate())
        else: pyg.draw.rect(scrn, b.color, b.coagulate())
        scrn.blit(b.text_surf(), b.text_coag())

def roll_one():
    to_sender = 1
    count = 0
    for i in xrange(10, 300, 10):
        count += 0.4
        to_sender = rand(1, 6)
        begin = pyg.time.get_ticks()
        while pyg.time.get_ticks() < begin + i:
            blit_all()
            scrn.blit(dies[to_sender], (425 + (math.sin(count) * 20), 425 + (math.cos(count) * 20)))
            pyg.display.update()
    begin = pyg.time.get_ticks()
    while pyg.time.get_ticks() < begin + 1000:
        blit_all()
        scrn.blit(dies[to_sender], (425 + (math.sin(count) * 20), 425 + (math.cos(count) * 20)))
        pyg.display.update()
    return to_sender

def roll_two():
    to_sender_a = 1
    to_sender_b = 1
    count = 0
    for i in xrange(10, 300, 10):
        count += 0.4
        to_sender_a = rand(1, 6)
        to_sender_b = rand(1, 6)
        begin = pyg.time.get_ticks()
        while pyg.time.get_ticks() < begin + i:
            blit_all()
            scrn.blit(dies[to_sender_a], (325 + (math.sin(count) * 20), 425 + (math.cos(count) * 20)))
            scrn.blit(dies[to_sender_b], (525 + (math.sin(-count) * 20), 425 + (math.cos(-count) * 20)))
            pyg.display.update()
    begin = pyg.time.get_ticks()
    while pyg.time.get_ticks() < begin + 1000:
        blit_all()
        scrn.blit(dies[to_sender_a], (325 + (math.sin(count) * 20), 425 + (math.cos(count) * 20)))
        scrn.blit(dies[to_sender_b], (525 + (math.sin(-count) * 20), 425 + (math.cos(-count) * 20)))
        pyg.display.update()
    return (to_sender_a, to_sender_b)

while True:

    blit_all()
    render_buttons()
    
    for ev in pyg.event.get():
        if ev.type == 12:
            pyg.quit()
            sys.exit()
        if ev.type == 5:
            if ev.button == 1:
                for b in buttons:
                    if b.isOver(pyg.mouse.get_pos()):
                        b.use_func(cply)

    pyg.display.update()
