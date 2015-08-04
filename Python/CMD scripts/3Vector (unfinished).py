import math

class Point:
    def __init__(this, x, y, z):
        for p in [x, y, z]:
            if type(p) != int:
                raise Exception("Point coordinates must be integers!")
        this.x = x
        this.y = y
        this.z = z
    
    def vect(this):
        return (this.x, this.y, this.z)

    def __str__(this):
        return str(this.vect())

    def __eq__(this, other):
        return this.vect() == other.vect()
    
    def _OP(this, other, oper):
        return Point(oper(this.x, other.x),
                     oper(this.y, other.y),
                     oper(this.z, other.z))
    
    def plus(this, other): 
        return this._OP(other, lambda a, b : a + b)
    def minus(this, other): 
        return this._OP(other, lambda a, b : a - b)
    def times(this, other): 
        return this._OP(other, lambda a, b : a * b)
    def div_by(this, other): 
        return this._OP(other, lambda a, b : a / b)
    
class Line:
    def __init__(this, point_a, point_b):
        for p in [point_a, point_b]:
            if not isinstance(p, Point):
                raise Exception("Lines must be defined with points!")
        if point_a == point_b:
            print "Points equal!"
            point_a.x += 1
        this.point_a = point_a
        this.point_b = point_b

    def slope(this):
        return this.point_a.minus(this.point_b)
        
    def length(this):
        a, b, c = this.slope().vect()
        return (a**2 + b**2 + c**2)**0.5

    def __str__(this):
        return str(this.point_a.vect()) + " to " + str(this.point_b.vect())

    def new_point(this, t):
        return this.slope().times(Point(t, t, t)).plus(this.point_a)

    def midpoint(this):
        return this.new_point(0.5)

    def intersect_plane(this, plane):
        pass

class Plane:
    def __init__(this, point_a, point_b, point_c):
        for p in [point_a, point_b, point_c]:
            if not isinstance(p, Point):
                raise Exception("Planes must be defined with points!")
        this.point_a = point_a
        this.point_b = point_b
        this.point_c = point_c

    def new_point(this, u, v):
        Pu = Point(u, u, u)
        Pv = Point(v, v, v)
        part1 = (this.point_b.minus(this.point_b)).times(Pu)
        part2 = (this.point_c.minus(this.point_b)).times(Pv)
        return this.point_a.plus(part1.plus(part2))
        

a = Point(2, 2, 2)
b = Point(4, 4, 4)
la = Line(a, b)

print a, b
print la
print la.length()
for x in xrange(-5, 5):
    print la.new_point(x)
