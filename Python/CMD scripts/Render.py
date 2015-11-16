import math
from PIL import Image
import time
import re

vertfetch_obj = re.compile("v ([^\s]*) ([^\s]*) ([^\s]*)")
facefetch_obj = re.compile("f ([^\s]*) ([^\s]*) ([^\s]*)")

class Point:
    def __init__(my, x, y, z):
        assert(type(x) in [int, float])
        assert(type(y) in [int, float])
        assert(type(z) in [int, float])
        my.x = float(x)
        my.y = float(y)
        my.z = float(z)

    def __str__(my):
        return "({0}, {1}, {2})".format(round(my.x, 2), round(my.y, 2), round(my.z, 2))

    def __repr__(my):
        return my.__str__()

    def __abs__(my):
        return math.sqrt(my.x**2 + my.y**2 + my.z**2)

    def __iter__(my):
        for p in [my.x, my.y, my.z]:
            yield p

    def __eq__(my, other):
        return my.x == other.x and my.y == other.y and my.z == other.z

    def __sub__(my, other):
        if type(other) == Point:
            return my.__sub(other)
        elif type(other) in [int, float]:
            return my.__sub(Point(other, other, other))
        else:
            return None
    def __sub(my, other):
        return Point(my.x - other.x, my.y - other.y, my.z - other.z)

    def __add__(my, other):
        if type(other) is Point:
            return my.__add(other)
        elif type(other) in [int, float]:
            return my.__add(Point(other, other, other))
        else:
            return None
    def __add(my, other):
        return Point(my.x + other.x, my.y + other.y, my.z + other.z)

    def __mul__(my, other):
        return Point(my.x * other, my.y * other, my.z * other)

    def __truediv__(my, other):
        return my.__div__(other)
    def __div__(my, other):
        return my * (1/other)

    def __neg__(my):
        return my * (-1)

    def distance(my, other):
        return abs(other - my)

    def dot(my, other):
        return my.x * other.x + my.y * other.y + my.z * other.z

    def cross(my, other):
        x = my.y * other.z - my.z * other.y
        y = my.z * other.x - my.x * other.z
        z = my.x * other.y - my.y * other.x
        return Point(x, y, z)

    def normalized(my):
        return my * 1/abs(my)

class Segment:
    def __init__(my, p1, p2):
        assert(type(p1) is Point)
        assert(type(p2) is Point)
        my.p1 = p1
        my.p2 = p2

        my.diff = my.calc_diff()

    def __str__(my):
        return "{0}-{1}".format(my.p1, my.p2)

    def calc_diff(my):
        my.diff = my.p2 - my.p1
        return my.diff

    def get_point(my, t):
        return my.p1 + my.diff * t

    def _raw_intersect(my, other):
        try:
            return -other.normal.dot(my.p1 - other.p1) / other.normal.dot(my.diff)
        except ZeroDivisionError:
            return None
        
    def _checktvalue(my, t):
        if t == None:
            return False
        return t >= 0 and t <= 1

    def clamped_intersection(my, other):
        if type(other) is Poly:
            tvalue = my._raw_intersect(other)
            if my._checktvalue(tvalue):
                u = other.p2 - other.p1
                v = other.p3 - other.p1
                w = my.get_point(tvalue) - other.p1
                uv = u.dot(v)
                uu = u.dot(u)
                wv = w.dot(v)
                wu = w.dot(u)
                vv = v.dot(v)
                denominator = uv*uv - uu*vv
                s = (uv*wv - vv*wu) / denominator
                if s < 0 or s > 1:
                    return None
                t = (uv*wu - uu*wv) / denominator
                if t >= 0 and s + t <= 1:
                    return tvalue
        return None

    def intersects(my, other):
        return my.clamped_intersection(other) != None

    def intersection(my, other):
        if type(other) is Poly:
            tvalue = my._raw_intersect(other)
            if tvalue == None: return None
            return my.get_point(tvalue)
        return False

    def _raw_closest(my, other):
        return -my.diff.dot(my.p1 - other) / my.diff.dot(my.diff)

    def closest(my, other):
        return my.get_point(my._raw_closest(other))

    def vectorize(my):
        t = abs(my)
        return my.p1, my.p2.diff.normalized, t

class Ray(Segment):
    def __str__(my):
        return "{0}->{1}".format(my.p1, my.p2)

    def _checktvalue(my, t):
        if t == None:
            return False
        return t >= 0
        
class Poly:
    def __init__(my, p1, p2, p3):
        assert(type(p1) is Point)
        assert(type(p2) is Point)
        assert(type(p3) is Point)
        my.p1 = p1
        my.p2 = p2
        my.p3 = p3

        my.normal = my.calc_normal()

    def __str__(my):
        return "{0}:{1}:{2}".format(my.p1, my.p2, my.p3)

    def __iter__(my):
        for p in [my.p1, my.p2, my.p3]:
            yield p

    def intersects(my, other):
        if type(other) in [Segment, Ray]:
            return other.intersects(my)
        return False

    def intersection(my, other):
        if type(other) in [Segment, Ray]:
            return other.intersection(my)
        return False

    def calc_normal(my):
        w = my.p3 - my.p1
        v = my.p2 - my.p1
        my.normal = w.cross(v).normalized()
        return my.normal

    def distance(my, other):
        return my.normal.dot(other - my.p1)

class Model:
    def __init__(my, param):
        my.tris = []
        if type(param) is list:
            my.tris = param
        elif type(param) is str:
            with open(param, "r") as f:
                if param.endswith(".obj"):
                    verts = []
                    for line in f:
                        if line.startswith("#"):
                            continue
                        elif line.startswith("v"):
                            m = re.search(vertfetch_obj, line)
                            verts.append(Point(float(m.group(1)), float(m.group(2)), float(m.group(3))))
                        elif line.startswith("f"):
                            m = re.search(facefetch_obj, line)
                            a, b, c = int(m.group(1)) - 1, int(m.group(2)) - 1, int(m.group(3)) - 1
                            my.tris.append(Poly(verts[a], verts[b], verts[c]))


class Point2d:
    def __init__(my, x, y, oldpoint=None):
        my.x = x
        my.y = y
        my.oldpoint = oldpoint

    def __sub__(my, other):
        return Point2d(my.x - other.x, my.y - other.y)

    def __add__(my, other):
        return Point2d(my.x + other.x, my.y + other.y)

    def __mul__(my, other):
        return Point2d(my.x * other, my.y * other)

    def __str__(my):
        return "({}, {})".format(my.x, my.y)

    def __repr__(my):
        return str(my)

    def cross(my, other):
        return (my.x * other.y) - (my.y * other.x)

    def clamped(my):
        newx = max(0, min(my.x, 1))
        newy = max(0, min(my.y, 1))
        return Point2d(newx, newy)

class Segment2d:
    def __init__(my, p1, p2, oldseg=None):
        my.p1 = p1
        my.p2 = p2
        my.oldseg = oldseg

        my.lower = Point2d(0, 0)
        my.upper = Point2d(0, 0)
        my.lower.x = min(p1.x, p2.x)
        my.lower.y = min(p1.y, p2.y)
        my.upper.x = max(p1.x, p2.x)
        my.upper.y = max(p1.y, p2.y)

        my.diff = my.p2 - my.p1

    def __str__(my):
        return "{}-{}".format(my.p1, my.p2)

    def __repr__(my):
        return str(my)

    def side(my, other):
        return (my.p2.x - my.p1.x) * (other.y - my.p1.y) - (my.p2.y - my.p1.y) * (other.x - my.p1.x)

    def intersection(my, other):
        numer = other.p1 - my.p1 
        denom = my.diff.cross(other.diff)
        t = numer.cross(other.diff) / denom
        #u = numer.cross(my.diff) / denom
        return my.get_point(t)

    def get_point(my, n):
        return my.p1 + my.diff * n

    def formula(my, x):
        if my.diff.x == 0:
            return my.p1.x
        slope = my.diff.y / my.diff.x
        return slope * (x - my.p1.x) + my.p1.y

class Poly2d:
    def __init__(my, p1, p2, p3, focal_analogue=None):
        my.p1 = p1
        my.p2 = p2
        my.p3 = p3
        my.focal_analogue = focal_analogue

        my.l1 = Segment2d(p1, p2)
        my.l2 = Segment2d(p2, p3)
        my.l3 = Segment2d(p3, p1)

        my.lower = Point2d(0, 0)
        my.upper = Point2d(0, 0)
        my.lower.x = min(p1.x, p2.x, p3.x)
        my.lower.y = min(p1.y, p2.y, p3.y)
        my.upper.x = max(p1.x, p2.x, p3.x)
        my.upper.y = max(p1.y, p2.y, p3.y)

        my.brange = Point2d(0, 0)
        my.brange.x = my.upper.x - my.lower.x
        my.brange.y = my.upper.y - my.lower.y

    def __str__(my):
        return "{}:{}:{}".format(my.p1, my.p2, my.p3)

    def __repr__(my):
        return str(my)

    def contains(my, other):
        return my.l1.side(other) >= 0 and my.l2.side(other) >= 0 and my.l3.side(other) >= 0



class Camera:
    def __init__(my, p1, p2, p3, focus, resx=100, resy=None):
        assert(type(p1) is Point)
        assert(type(p2) is Point)
        assert(type(p3) is Point)
        assert(type(focus) in [Point, float, int])
        assert(type(resx) is int)
        my.p1 = p1
        my.p2 = p2
        my.p3 = p3
        my.resx = resx

        my.cross_segment = Segment(p2, p3)
        if resy == None:
            mult = my.p1.distance(my.p2) / my.p1.distance(my.p3)
            my.resy = int(my.resx / mult)
        else:
            my.resy = resy
        midpoint = my.cross_segment.get_point(0.5)
        my.p4 = Segment(p1, midpoint).get_point(2)
        my.poly1 = Poly(my.p1, my.p2, my.p3)
        my.poly2 = Poly(my.p3, my.p2, my.p4)
        my.buff = FrameBuffer(my.resx, my.resy)
        my.pcount = my.resx * my.resy
        if type(focus) is Point:
            my.focus = focus
        else:
            my.focus = midpoint + my.poly1.normal * focus

    def get_realpos(my, pt, from_pixel=True):
        leftrail = Segment(my.p1, my.p3)
        rightrail = Segment(my.p2, my.p4)
        ydist = pt.y
        xdist = pt.x
        if from_pixel:
            ydist = pt.y / my.resy 
            xdist = pt.x / my.resx
        ladderstep = Segment(leftrail.get_point(ydist), rightrail.get_point(ydist))
        return ladderstep.get_point(xdist)

    def intersects(my, other):
        return my.poly1.intersects(other) or my.poly2.intersects(other)

    def intersection(my, other):
        return my.poly1.intersection(other)

    def flush(my, filename, track_progress=False):
        my.buff.flush(filename, track_progress)
        my.buff.clear()

    def _raw_render(my, other, shader=lambda d, pol, pix: (255, 255, 255), track_progress=False, ignore_z=False, new_depth_test=False):
        pass

    def render(my, other, shader=lambda d, pol, pix: (255, 255, 255), track_progress=False, ignore_z=False, new_depth_test=False):
        l = []
        if type(other) is Model:
            counter = PercentCounter(len(other.tris)-1)
            for i, tri in enumerate(other.tris):
                l += list(my._raw_render(tri, shader, track_progress, ignore_z, new_depth_test))
                counter.incr()
        else:
            return list(my._raw_render(other, shader, track_progress, ignore_z, new_depth_test))
        return l


class RayTracer(Camera):
    def __iter__(my):
        leftrail = Segment(my.p1, my.p3)
        rightrail = Segment(my.p2, my.p4)
        yincr = 1/my.resy
        xincr = 1/my.resx
        y, yi = 0, 0
        while yi < my.resy:
            x, xi = 0, 0
            ladderstep = Segment(leftrail.get_point(y), rightrail.get_point(y))
            while xi < my.resx:
                yield (xi, yi), ladderstep.get_point(x)
                x += xincr
                xi += 1
            y += yincr
            yi += 1

    def fetch_rays(my):
        for (x, y), pixel in my:
            yield (x, y), Ray(pixel, my.focus)
    
    def render_skybox(my, gcol, scol):
        percount = PercentCounter(my.pcount)
        gplane = Poly(
                Point(0, 999, 0),
                Point(1, 999, 0),
                Point(0, 999, 1)
            )
        splane = Poly(
                Point(0, -999, 0),
                Point(1, -999, 0),
                Point(0, -999, 1)
            )
        for (x, y), ray in my.fetch_rays():
            if ray._raw_intersect(splane) > 0:
                my.buff.imprint_ignorezbuffer(x, y, gcol)
            if ray._raw_intersect(gplane) > 0:
                my.buff.imprint_ignorezbuffer(x, y, scol)
            #percount.incr()

    def _raw_render(my, other, shader=lambda d, pol, pix: (255, 255, 255), track_progress=False, ignore_z=False, old_depth_test=False):
        counter = PercentCounter(my.pcount)
        if type(other) is Poly:
            for (x, y), ray in my.fetch_rays():
                if track_progress:
                    counter.incr()
                t = ray.clamped_intersection(other)
                if t != None:
                    my.buff.imprint(x, y, shader(t, ray, (x, y)), t)

class Rasterizer(Camera):
    def get_2d_point(my, other):
        ray = Ray(my.focus, other)
        collision = ray.intersection(my.poly1)
        x = Segment(my.p1, my.p2)._raw_closest(collision)
        y = Segment(my.p1, my.p3)._raw_closest(collision)
        return Point2d(x, y, other)

    def flatten(my, other):
        if type(other) is Poly:
            p1 = my.get_2d_point(other.p1)
            p2 = my.get_2d_point(other.p2)
            p3 = my.get_2d_point(other.p3)
            return Poly2d(p1, p2, p3, other)
        elif type(other) is Segment:
            p1 = my.get_2d_point(other.p1)
            p2 = my.get_2d_point(other.p2)
            return Segment2d(p1, p2, other)
        elif type(other) is Point:
            return my.get_2d_point(other)

    def pixelate(my, other):
        newx = int(other.x * my.resx)
        newy = int(other.y * my.resy)
        return Point2d(newx, newy)

    def fetch_segment_slices(my, other):
        flist = []
        bottom = my.pixelate(other.lower)
        top = my.pixelate(other.upper)
        diff = other.oldseg.p2 - other.oldseg.p1

        lowreal = my.get_realpos(other.p1)
        highreal = my.get_realpos(other.p2)
        lowtvalue = lowreal.distance(other.oldseg.p1) / lowreal.distance(my.focus)
        hightvalue = highreal.distance(other.oldseg.p2) / highreal.distance(my.focus)
        tdiff = hightvalue - lowtvalue
        startdist = lowtvalue

        lengthrange = range(max(bottom.x, 0), min(top.x+1, my.resx))
        incr = 1/len(lengthrange) * tdiff
        t = lowtvalue

        for x in lengthrange:
            t += incr
            flist.append((x, x / my.resx, t))
        return flist

    def fetch_boundbox_pixels(my, other):
        flist = []
        pup = my.pixelate(other.upper)
        pdn = my.pixelate(other.lower)

        downleft_corner = Point2d(other.upper.x, other.lower.y)
        upright_corner = Point2d(other.lower.x, other.upper.y)

        upleft_raycast = Ray(my.get_realpos(other.upper, False), my.focus)
        downright_raycast = Ray(my.get_realpos(other.lower, False), my.focus)
        downleft_raycast = Ray(my.get_realpos(downleft_corner, False), my.focus)
        upright_raycast = Ray(my.get_realpos(upright_corner, False), my.focus)

        upleft_tvalue = upleft_raycast._raw_intersect(other.focal_analogue)
        downright_tvalue = downright_raycast._raw_intersect(other.focal_analogue)
        downleft_tvalue = downleft_raycast._raw_intersect(other.focal_analogue)
        upright_tvalue = upright_raycast._raw_intersect(other.focal_analogue)

        left_diff =  upleft_tvalue - downleft_tvalue
        right_diff = upright_tvalue - downright_tvalue

        x_range = range(max(pdn.x, 0), min(pup.x+1, my.resx))
        y_range = range(max(pdn.y, 0), min(pup.y+1, my.resy))
        if len(x_range) == 0 or len(y_range) == 0:
            return []

        x_incr = 1/len(x_range)
        y_incr = 1/len(y_range)

        for yi, y in enumerate(y_range):
            yv = y_incr * yi
            yv = math.log(yv + 1, 2)
            l_tvalue = left_diff * yv + downleft_tvalue
            r_tvalue = right_diff * yv + downright_tvalue
            t_diff = r_tvalue - l_tvalue
            for xi, x in enumerate(x_range):
                xv = (xi * x_incr)
                t = t_diff * xv + r_tvalue
                flist.append((Point2d(x, y), Point2d(x / my.resx, y / my.resy), t, (xi, yi)))
        return flist

    def _raw_render(my, other, shader=lambda d, pol, pix: (255, 255, 255), track_progress=False, ignore_z=False, new_depth_test=False):
        if type(other) is Poly:
            yield -1, -1, str(other)
            pl = my.flatten(other)
            counter = PercentCounter(pl.brange.x * pl.brange.y * my.resx * my.resy)
            for pixel, rpos, t, xy in my.fetch_boundbox_pixels(pl):
                x, y = xy
                if track_progress:
                    counter.incr()
                if pl.contains(rpos):
                    if ignore_z:
                        my.buff.imprint_ignorezbuffer(pixel.x, pixel.y, shader(0, other, pixel))
                    else:
                        if not new_depth_test:
                            t = Ray(my.get_realpos(pixel), my.focus)._raw_intersect(other)
                        yield pixel.x, pixel.y, t
                        my.buff.imprint(pixel.x, pixel.y, shader(t, other, pixel), t)
        if type(other) is Segment:
            sg = my.flatten(other)
            for x, rx, t in my.fetch_segment_slices(sg):
                if track_progress:
                    counter.incr()
                ry = sg.formula(rx)
                pixel = my.pixelate(Point2d(rx, ry))
                if ignore_z:
                    my.buff.imprint_ignorezbuffer(pixel.x, pixel.y, shader(0, other, pixel))
                else:
                    my.buff.imprint(pixel.x, pixel.y, shader(t, other, pixel), t)

class FrameBuffer:
    def __init__(my, wid, hih):
        assert(type(wid) is int)
        assert(type(hih) is int)
        my.wid = wid
        my.hih = hih
        my.im = None
        my.dim = (my.wid, my.hih)
        my.buff = [[(0, 0, 0) for i in range(hih)] for j in range(wid)]
        my.zbuff = [[-1 for i in range(hih)] for j in range(wid)]

    def flush(my, filename, track_progress=False):
        counter = PercentCounter(my.dim[0] * my.dim[1])
        my.im = Image.new("RGB", my.dim)
        for xi, y in enumerate(my.buff):
            for yi, x in enumerate(y):
                if track_progress:
                    counter.incr()
                my.im.putpixel((my.wid - xi - 1, my.hih - yi - 1), tuple(int(v) % 255 for v in x))
        my.im.save(filename, "PNG")

    def view(my):
        if my.im == None:
            print("no image to show")
        else:
            my.im.show()

    def clear(my):
        for y in range(my.hih):
            for x in range(my.wid):
                my.buff[x][y] = (0, 0, 0)
                my.zbuff[x][y] = -1

    def fill(my, val):
        assert(type(val) is tuple)
        assert(len(val) is 3)
        for y in range(my.hih):
            for x in range(my.wid):
                my.buff[x][y] = val

    def imprint(my, x, y, val, zval):
        assert(type(val) is tuple)
        assert(len(val) is 3)
        if my.zbuff[x][y] == -1 or my.zbuff[x][y] >= zval:
            my.buff[x][y] = val
            my.zbuff[x][y] = zval

    def imprint_ignorezbuffer(my, x, y, val):
        assert(type(val) is tuple)
        assert(len(val) is 3)
        my.buff[x][y] = val


class PercentCounter:
    def __init__(my, maxnum):
        my.maxnum = maxnum

        my.increment = 0
        my.goalnum = 0
        my.starttime = time.clock()
        my.endditme = -1
        my.done = False

    def incr(my, amount=1.0):
        if my.done:
            return
        my.increment += amount
        percent = my.increment/my.maxnum * 100
        if percent > my.goalnum:
            print("{}%".format(my.goalnum))
            if my.goalnum >= 100:
                my.endtime = time.clock()
                my.done = True
                print("Took {} seconds.".format(round(my.endtime - my.starttime, 3)))
            my.goalnum = math.ceil(percent)

cameras = {
    'buncam_rs' : Rasterizer(
        Point(-0.1, 0, -0.19),
        Point(0.1, 0, -0.19),
        Point(-0.1, 0.2, -0.19),
        0.5, 1000
    ),
    'sidecam_rs' : Rasterizer(
        Point(-1, 5, -4),
        Point(1, 5, -4),
        Point(-1, 4, -5),
        2, 500
    )
}

'''
    'topcam_rs' : Rasterizer(
        Point(-1, 3, 1),
        Point(1, 3, 1),
        Point(-1, 3, -1),
        1, 500
    ),
    'isocam_rs' : Rasterizer(
        Point(2, 5, -3),
        Point(3, 5, -2),
        Point(3, 4, -4),
        1, 500
    ),
    'buncam2_rs' : Rasterizer(
        Point(0.1, 0, 0.5),
        Point(-0.1, 0, 0.5),
        Point(0.1, 0.2, 0.5),
        1, 500
    ),
    'topcam_rt' : RayTracer(
        Point(-1, 3, 1),
        Point(1, 3, 1),
        Point(-1, 3, -1),
        1, 500
    ),

    'isocam_rt' : RayTracer(
        Point(2, 5, -3),
        Point(3, 5, -2),
        Point(3, 4, -4),
        1, 500
    ),

    'sidecam_rt' : RayTracer(
        Point(-1, 5, -4),
        Point(1, 5, -4),
        Point(-1, 4, -5),
        2, 500
    ),

    'undercam_rt' : RayTracer(
        Point(-1, -4, -4),
        Point(1, -4, -4),
        Point(-1, -5, -3),
        1, 500
    ),
    'undercam_rs' : Rasterizer(
        Point(-1, -4, -4),
        Point(1, -4, -4),
        Point(-1, -5, -3),
        1, 500
    )
'''

def print_table(t, end="\n"):
    for row in t:
        for item in row:
            if item == (0, 0, 0):
                print(".", end=" ")
            else:
                print("0", end=" ")
        print()
    print(end, end="")

factor = 1.5
def logshader(d, pol, pix):
    v = 255 / math.log(max(-d, 0) + factor, factor)
    if type(pol) is Poly:
        return v, v, v
    else:
        return v, 0, 0

mult = 2000
def basicshader(d, pol, pix):
    v = d*mult
    return v, v, v

loc = Point(0, -1, 0)
def dirshader(d, pol, pix):
    v = pol.normal.dot(loc) * 127 + 128
    return v, v, v

csec = 335
def markshader(d, pol, pix):
    v = d*mult
    if pix.x == csec:
        return v, 0, 0
    return v, v, v

def main():
    import os
    storedir = "renders"
    if not os.path.isdir(storedir):
        os.mkdir(storedir)

    print("loading model")
    bunbun = Model("models/bunny.obj")
    sampletris = Model("models/testpolys.obj")

    # 1st pass
    cname = "sidecam_rs"
    print("rendering camera " + cname + " with new depth pass model")
    cam = cameras[cname]
    start = time.clock()
    print("tracing background")
    cam.buff.fill((32, 32, 32))
    items = cam.render(sampletris, markshader, new_depth_test=True)
    with open(storedir + "/new.txt", "w") as f:
        for x, y, n, in items:
            if x == csec:
                f.write("{},{},\n".format(y, n))

    print("Total render time: {} seconds".format(round(time.clock() - start, 3)))
    print("Flushing results")
    cam.flush(storedir + "/{}_new.png".format(cname))

    # 2nd pass
    print("rendering camera " + cname + " with old depth pass model")
    start = time.clock()
    print("tracing background")
    cam.buff.fill((32, 32, 32))
    items = cam.render(sampletris, markshader, new_depth_test=False)
    with open(storedir + "/old.txt", "w") as f:
        for x, y, n in items:
            if x == csec:
                f.write("{},{},\n".format(y, n))
    print("Total render time: {} seconds".format(round(time.clock() - start, 3)))
    print("Flushing results")
    cam.flush(storedir + "/{}_old.png".format(cname))

if __name__ == "__main__":
    main()