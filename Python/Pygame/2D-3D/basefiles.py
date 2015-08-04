import pygame as pyg
import sys
import math

white = (255, 255, 255)
light = (196, 196, 196)
grey = (128, 128, 128)
dark = (64, 64, 64)
black = (0, 0, 0)
red = (230, 40, 40)
reddark = (120, 10, 10)
blue = (40, 40, 230)
purple = (196, 20, 196)

def sectionrender(scrn, berth, size, color=white):
    pyg.draw.rect(scrn, color, [berth[0], 0, berth[1] - berth[0], size.y])

class Point:
    def __init__(this, x, y):
        this.x = x # Float
        this.y = y # Float

    def listify(this):
        return [this.x, this.y]
    
    def render(this, scrn, color=white):
        pyg.draw.rect(scrn, color, [int(this.x)-1, int(this.y)-1, 3, 3])

class Line:
    def __init__(this, pos1, pos2):
        this.pos1 = pos1 # Point
        this.pos2 = pos2 # Point

    def length(this):
        dx = this.pos1.x - this.pos2.x
        dy = this.pos1.y - this.pos2.y
        return (dx**2 + dy**2)**0.5

    def slope(this):
        try: return float(this.pos1.y - this.pos2.y) / float(this.pos1.x - this.pos2.x)
        except ZeroDivisionError:
            return float(this.pos1.y - this.pos2.y) / 0.0001

    def yIntercept(this):
        return this.pos1.y - this.slope() * this.pos1.x
    
    def side(this, point):
        a = (this.pos2.x - this.pos1.x) * (point.y - this.pos2.y)
        b = (this.pos2.y - this.pos1.y) * (point.x - this.pos2.x)
        return a - b > 0

    def inbox(this, point):
        thisminx, thismaxx = min(this.pos1.x, this.pos2.x), max(this.pos1.x, this.pos2.x)
        thisminy, thismaxy = min(this.pos1.y, this.pos2.y), max(this.pos1.y, this.pos2.y)
        if thisminx > point.x: return False
        if thismaxx < point.x: return False
        if thisminy > point.y: return False
        if thismaxy < point.y: return False
        return True
        
    def intersects(this, other):
        if type(other) != type(this):
            raise Exception('Argument must be a line.')
        if this.side(other.pos1) == this.side(other.pos2): return False
        if other.side(this.pos1) == other.side(this.pos2): return False
        return True

    def intersection(this, other):
        a = this.yIntercept() - other.yIntercept()
        b = this.slope() - other.slope()
        x = a / b
        return Point(x * (-1), ((x * this.slope()) - this.yIntercept()) * (-1))

    def render(this, scrn, color=grey):
        pyg.draw.line(scrn, color, [int(this.pos1.x),
                                    int(this.pos1.y)],
                                   [int(this.pos2.x),
                                    int(this.pos2.y)])
    
class Camera:
    def __init__(this, pos, ang, renders=[]):
        this.pos = pos # Point
        this.ang = ang # int
        this.renders = renders # Renderables

    def getsin(this): return  math.sin((math.pi * this.ang) / 180.0 + (math.pi/4))
    def getcos(this): return  math.cos((math.pi * this.ang) / 180.0 + (math.pi/4))
    
    def render(this, scrn, color=blue):
        scrnpos1, scrnpos2 = Point(0, 0), Point(0, 0)
        scrnpos1.x = math.sin((math.pi * this.ang) / 180.0) * 30 + this.pos.x
        scrnpos1.y = math.cos((math.pi * this.ang) / 180.0) * 30 + this.pos.y
        scrnpos2.x = math.sin((math.pi * this.ang) / 180.0 + (math.pi/2)) * 30 + this.pos.x
        scrnpos2.y = math.cos((math.pi * this.ang) / 180.0 + (math.pi/2)) * 30 + this.pos.y
        screenline = Line(scrnpos1, scrnpos2)
        sideline1 = Line(this.pos, scrnpos1)
        sideline2 = Line(this.pos, scrnpos2)
        
        screenline.render(scrn, color)
        sideline1.render(scrn, grey)
        sideline2.render(scrn, grey)
        this.pos.render(scrn, color)

        for r in this.renders:
            if isinstance(r, Point):
                viewline = Line(r, this.pos)
                if viewline.intersects(screenline):
                    viewline.intersection(screenline).render(scrn, white)
            if isinstance(r, Line):
                viewline1 = Line(r.pos1, this.pos)
                viewline2 = Line(r.pos2, this.pos)
                c, d = False, False
                if not r.side(this.pos):
                    if sideline1.side(r.pos1) and not sideline1.side(r.pos2):
                        c = True
                    if sideline2.side(r.pos1) and not sideline2.side(r.pos2):
                        d = True
                else:
                    if not sideline1.side(r.pos1) and sideline1.side(r.pos2):
                        c = True
                    if not sideline2.side(r.pos1) and sideline2.side(r.pos2):
                        d = True
                if c and d:
                    screenline.render(scrn, white)           
                a = viewline1.intersects(screenline) 
                b = viewline2.intersects(screenline)
                if a and b:
                    Line(viewline1.intersection(screenline),
                         viewline2.intersection(screenline)).render(scrn, white)
                if a and not b:
                    if r.side(this.pos):
                        Line(viewline1.intersection(screenline),
                             scrnpos1).render(scrn, white)
                    else:
                        Line(viewline1.intersection(screenline),
                             scrnpos2).render(scrn, white)
                if b and not a:
                    if r.side(this.pos):
                        Line(viewline2.intersection(screenline),
                             scrnpos2).render(scrn, white)
                    else:
                        Line(viewline2.intersection(screenline),
                             scrnpos1).render(scrn, white)
            
    def render3D(this, scrn, size, color=blue):
        scrnpos1, scrnpos2 = Point(0, 0), Point(0, 0)
        scrnpos1.x = math.sin((math.pi * this.ang) / 180.0) * 30 + this.pos.x
        scrnpos1.y = math.cos((math.pi * this.ang) / 180.0) * 30 + this.pos.y
        scrnpos2.x = math.sin((math.pi * this.ang) / 180.0 + (math.pi/2)) * 30 + this.pos.x
        scrnpos2.y = math.cos((math.pi * this.ang) / 180.0 + (math.pi/2)) * 30 + this.pos.y
        screenline = Line(scrnpos1, scrnpos2)
        sideline1 = Line(this.pos, scrnpos1)
        sideline2 = Line(this.pos, scrnpos2)

        maxlen = screenline.length()

        for r in this.renders:
            if isinstance(r, Point):
                viewline = Line(r, this.pos)
                if viewline.intersects(screenline):
                    viewline.intersection(screenline).render(scrn, white)
            if isinstance(r, Line):
                viewline1 = Line(r.pos1, this.pos)
                viewline2 = Line(r.pos2, this.pos)
                c, d = False, False
                if not r.side(this.pos):
                    if sideline1.side(r.pos1) and not sideline1.side(r.pos2):
                        c = True
                    if sideline2.side(r.pos1) and not sideline2.side(r.pos2):
                        d = True
                else:
                    if not sideline1.side(r.pos1) and sideline1.side(r.pos2):
                        c = True
                    if not sideline2.side(r.pos1) and sideline2.side(r.pos2):
                        d = True
                if c and d:
                    sectionrender(scrn, [0, size.x], size)
                    
                a = viewline1.intersects(screenline) 
                b = viewline2.intersects(screenline)
                rline = Line(scrnpos1, scrnpos1)
                if a and b:
                    rline = Line(viewline1.intersection(screenline),
                         viewline2.intersection(screenline))
                elif a and not b:
                    if r.side(this.pos):
                        rline = Line(viewline1.intersection(screenline),
                             scrnpos1)
                    else:
                        rline = Line(viewline1.intersection(screenline),
                             scrnpos2)
                elif b and not a:
                    if r.side(this.pos):
                        rline = Line(viewline2.intersection(screenline),
                             scrnpos2)
                    else:
                        rline = Line(viewline2.intersection(screenline),
                             scrnpos1)
                berth = [
                    Line(scrnpos1, rline.pos1).length() / maxlen,
                    Line(scrnpos1, rline.pos2).length() / maxlen]
                berth = [x * size.x for x in berth]
                sectionrender(scrn, berth, size)
                        
                
if __name__ == '__main__': print "\n\t\t\tWrong module, dummy.\n"
