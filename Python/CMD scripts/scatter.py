import numpy as np
import cv2
import math
import cmath
import random

def plot(eq, dim=(1280, 1280), scale=1):
  im = np.zeros(dim)
  width, height = dim
  for i in range(500):
    x, y = eq()
    gx = int((x - 1) * width)
    gy = int((y - 1) * height)
    if gx < 0 or gx >= width or gy < 0 or gy >= height:
      continue
    im[gx][gy] = 1
  cv2.imshow("plot", im)
  cv2.waitKey()

def randomperim():
  ntype = lambda: 1 if random.random() > 0.5 else -1
  a = random.random() * 2 - 1;
  b = 1 - (a*a);
  b = math.sqrt(b) * ntype()
  return a, b

def perim2():
  n = cmath.exp(1 + random.random() * math.pi * 2j)
  return n.real, n.imag

plot(randomperim)
plot(perim2)