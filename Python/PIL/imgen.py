import numpy as np
import cv2
import math
import cmath
import os

def evaldraw(eq, size=(1280, 1280), *args):
  width, height = size
  imdat = np.zeros((height, width, 3))
  ratio = width / height
  for x in range(width):
    for y in range(height):
      relx = x / width
      rely = y / height
      if len(args) > 0:
        imdat[y][x] = [min(max(0, n), 1) for n in eq((relx, rely), ratio, *args)]
      else:
        imdat[y][x] = [min(max(0, n), 1) for n in eq((relx, rely), ratio)]
  return imdat

def sigmoid(x, modif=6):
  return 1 / (1 + math.exp(modif*(x - 0.5)))

def deltasigmoid(x):
  return 4 * (math.exp(-x) / (1 + math.exp(-x)) ** 2)

def modifiedsine(x, modif=4):
  zig = abs((x % 8) - 4) - 2
  return 2 / (1 + math.exp(-(modif*zig))) - 1

def distance(x1, y1, x2, y2):
  dx, dy = x2-x1, y2-y1
  return math.sqrt(dx*dx + dy*dy)

def rotabout(x, y, xa, ya, ang):
  cang, sang = math.cos(ang), math.sin(ang)
  dx, dy = x - xa, y - ya
  nx = cang*dx - sang*dy
  ny = sang*dx + cang*dy
  return nx + xa, ny + ya

def grid(pos, ratio):
  x, y = pos
  x *= ratio
  rx, ry = x-ratio/2, y-0.5
  if rx*rx + ry*ry < 0.1:
    rx, ry = rotabout(rx, ry, 0, 0, 0.25)
    rx, ry = rx*0.4, ry*0.4
    x, y = rx + ratio / 2, ry + 0.5
  gsize = 20
  bgpx, bgpy = int(x*gsize)/gsize, int(y*gsize)/gsize
  m = 0
  for gpx, gpy in [(a, b) for a in [bgpx, bgpx+(1/gsize)] for b in [bgpy, bgpy+(1/gsize)]]:
    m = max(m, deltasigmoid(distance(x, y, gpx, gpy) * gsize * 12))
  m /= 3
  if (int(x*gsize)%2 == 0) ^ (int(y*gsize)%2 == 0):
    m = 1 - m
  return [m]*3

def blurgrid(pos, ratio):
  x, y = pos
  x *= ratio
  return [modifiedsine(x*50) * modifiedsine(y*50)]*3

def spiral(pos, ratio):
  x, y = pos
  x *= ratio
  dist = distance(x, y, ratio / 2, 0.5)
  rx, ry = rotabout(x, y, ratio / 2, 0.5, dist*30)
  return [deltasigmoid((0.5 - ry) / max(0.005, dist/4))]*3

def cspiral(pos, ratio, modif=1):
  x, y = pos
  cnt = ratio / 2 + 0.5j
  cpx = x + y*1j
  cpx -= cnt
  unt = cpx / max(0.01, abs(cpx))
  cpx *= cmath.rect(1, abs(cpx)*modif)
  angdist = (cmath.phase(0j - 1) - cmath.phase(cpx)) / cmath.pi
  if angdist > 1:
    angdist = 2 - angdist
  return [sigmoid(angdist)]*3

def dstest(pos, ratio):
  x, y = pos
  x *= ratio
  return [deltasigmoid(distance(x, y, 0.5, 0.5)*16)]*3

# evals = [
#   cspiral
# ]

# results = []
# for e in evals:
#   print("evaluating " + e.__name__)
#   results.append((e, evaldraw(e, (720, 720))))

# print("showing")

# for f, r in results:
#   cv2.imshow("result of " + f.__name__, r)
#   cv2.imwrite(f.__name__ + ".png", r*255)

# cv2.waitKey()

outdir = "output-two"

if not os.path.isdir(outdir):
  os.mkdir(outdir)

for i in range(960):
  print("generating frame {}".format(i+1))
  cv2.imwrite(outdir + "/animframe_{}.png".format(i+1), evaldraw(cspiral, (720, 720), (i/16)**2)*255)

# for i in range(64):
#   x = i/4 - 8
#   print("#"*int(deltasigmoid(x)*100) + " {} ->  {}".format(x, deltasigmoid(x)))