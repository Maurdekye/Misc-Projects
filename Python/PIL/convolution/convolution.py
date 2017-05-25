import os, sys
import numpy as np
import cv2
import math

if len(sys.argv) < 2:
  input("Please give image file argument")
  sys.exit()

impath = sys.argv[1]

if not os.path.isfile(impath):
  input("File not exist")
  sys.exit()

im = cv2.imread(impath, cv2.IMREAD_UNCHANGED)

if im is None:
  input("Invalid image file")
  sys.exit()

# do convolution

def sigmoid(x: float) -> float:
  if abs(x) > 100:
    return x / abs(x)
  return 1 / (1 + math.exp(-x))

def kernel_at(base: np.ndarray, kernel: np.ndarray, x: int, y: int) -> float:
  kw, kh = kernel.shape
  halfx, halfy = kw // 2, kh // 2
  ksum = 0
  for kx in range(kw):
    for ky in range(kh):
      xd, yd = kx - halfx, ky - halfy
      ksum += kernel[kx,ky] * base[x + xd,y + yd]
  return ksum

def convolute(image: np.ndarray, kernel: np.ndarray, *, debugprogress=False) -> np.ndarray:
  kw, kh = kernel.shape
  channels = cv2.split(image)
  newwid = image.shape[0] - kw
  newhih = image.shape[1] - kh
  newchannels = []
  totalprog = newwid * newhih * len(channels)
  progcounter = 0
  percentcounter = 0
  for channel in channels:
    newchannel = np.array([[0] * newhih] * newwid)
    xbuff, ybuff = kw // 2, kh // 2
    for x in range(newwid):
      for y in range(newhih):
        kresult = kernel_at(channel, kernel, x + xbuff, y + ybuff)
        pixel = int((sigmoid(kresult) + 1) * 127)
        newchannel[x,y] = pixel
        progcounter += 1
        if progcounter > percentcounter:
          percentcounter += totalprog // 100
          print(f"{(progcounter * 100) // totalprog}% finished")
    newchannels.append(newchannel)

  return cv2.merge(newchannels)

kernels = {
  "vertical_edge" : np.array([
    [-0.7, -0.2, 0.0, 0.2, 0.7],
    [-0.7, -0.2, 0.0, 0.2, 0.7],
    [-0.7, -0.2, 0.0, 0.2, 0.7],
    [-0.7, -0.2, 0.0, 0.2, 0.7],
    [-0.7, -0.2, 0.0, 0.2, 0.7]]),
  "tall_vertical_edge" : np.array([
    [ 1.0,  1.0,  1.0],
    [ 0.7,  0.7,  0.7],
    [ 0.2,  0.2,  0.2],
    [ 0.0,  0.0,  0.0],
    [-0.2, -0.2, -0.2],
    [-0.7, -0.7, -0.7],
    [-1.0, -1.0, -1.0]]),
  "very_tall_vertical_edge" : np.array([
    [1.5], [1.4], [1.2], [1.0], [0.7], [0.5], [0.2], [0.0], 
    [-0.2], [-0.5], [-0.7], [-1.0], [-1.2], [-1.4], [-1.5]]),
  "horizontal_edge" : np.array([
    [ 0.7,  0.7,  0.7,  0.7,  0.7],
    [ 0.2,  0.2,  0.2,  0.2,  0.2],
    [ 0.0,  0.0,  0.0,  0.0,  0.0],
    [-0.2, -0.2, -0.2, -0.2, -0.2],
    [-0.7, -0.7, -0.7, -0.7, -0.7]]),
  "long_horizontal_edge" : np.array([
    [ 1.0,  0.7,  0.2,  0.0, -0.2, -0.7, -1.0],
    [ 1.0,  0.7,  0.2,  0.0, -0.2, -0.7, -1.0],
    [ 1.0,  0.7,  0.2,  0.0, -0.2, -0.7, -1.0]]),
  "very_long_horizontal_edge" : np.array([
    [1.5, 1.4, 1.2, 1.0, 0.7, 0.5, 0.2, 0.0, -0.2, -0.5, -0.7, -1.0, -1.2, -1.4, -1.5]]),
  "huge_horizontal_edge" : np.array([
    [1.5, 1.4, 1.2, 1.0, 0.7, 0.5, 0.2, 0.0, -0.2, -0.5, -0.7, -1.0, -1.2, -1.4, -1.5],
    [1.5, 1.4, 1.2, 1.0, 0.7, 0.5, 0.2, 0.0, -0.2, -0.5, -0.7, -1.0, -1.2, -1.4, -1.5],
    [1.5, 1.4, 1.2, 1.0, 0.7, 0.5, 0.2, 0.0, -0.2, -0.5, -0.7, -1.0, -1.2, -1.4, -1.5],
    [1.5, 1.4, 1.2, 1.0, 0.7, 0.5, 0.2, 0.0, -0.2, -0.5, -0.7, -1.0, -1.2, -1.4, -1.5],
    [1.5, 1.4, 1.2, 1.0, 0.7, 0.5, 0.2, 0.0, -0.2, -0.5, -0.7, -1.0, -1.2, -1.4, -1.5],
    [1.5, 1.4, 1.2, 1.0, 0.7, 0.5, 0.2, 0.0, -0.2, -0.5, -0.7, -1.0, -1.2, -1.4, -1.5],
    [1.5, 1.4, 1.2, 1.0, 0.7, 0.5, 0.2, 0.0, -0.2, -0.5, -0.7, -1.0, -1.2, -1.4, -1.5]]),
  "tiny_edge" : np.array([[-1.0, 1.0]]),
  "spot" : np.array([
    [-0.8, -0.4, -0.1, -0.4, -0.8],
    [-0.4,  0.6,  1.8,  0.6, -0.4],
    [-0.1,  1.8,  2.4,  1.8, -0.1],
    [-0.4,  0.6,  1.8,  0.6, -0.4],
    [-0.8, -0.4, -0.1, -0.4, -0.8]])
  }

active_kernel = "vertical_edge"

if len(sys.argv) >= 3:
  active_kernel = sys.argv[2]

if active_kernel not in kernels:
  input(f"{active_kernel} is not an availble kernel")
  sys.exit()

print("Performing convolution...")
newim = convolute(im, kernels[active_kernel]*0.1, debugprogress=True)
print("Saving to file")
newname = impath[:1-impath.rfind(".")] + f"_convoluted_{active_kernel}.jpg"
cv2.imwrite(newname, newim)
print("Finished.")