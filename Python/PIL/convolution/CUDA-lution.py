import pycuda.autoinit
import pycuda.driver as device
from pycuda.compiler import SourceModule as cpp

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

factor = 1.0
if len(sys.argv) >= 4:
  factor = float(sys.argv[3])

# do convolution

proc_kernel = cpp("""
__global__ void proc_kernel(float* factor, int* imwidth, int* imheight, int* kernelwidth, int* kernelheight, unsigned char* inputimage, float* kernel, unsigned char* outputimage)
{
  const int id = threadIdx.x + blockIdx.x * blockDim.x;
  int outputwidth = *imwidth - *kernelwidth;
  int outputheight = *imheight - *kernelheight;

  if (id >= outputwidth * outputheight)
    return;

  int x = id % outputwidth;
  int y = id / outputwidth;

  int halfkernelwidth = *kernelwidth / 2;
  int halfkernelheight = *kernelheight / 2;

  float sum = 0.0f;

  for (int kx = 0; kx < *kernelwidth; kx++)
  {
    for (int ky = 0; ky < *kernelheight; ky++)
    {
      int xd = kx - halfkernelwidth;
      int yd = ky - halfkernelheight;
      int newx = x + xd;
      int newy = y + yd;

      sum += kernel[kx + ky * *kernelwidth] * inputimage[newx + newy * *imwidth];
    }
  }

  float sigmoid;
  float fsum = sum;
  fsum *= *factor;
  if (fabsf(fsum) > 32)
    sigmoid = sum / fabsf(fsum);
  else
    sigmoid = 1.0f / (1.0f + expf(-fsum));

  int pixelcolor = (sigmoid + 1) * 127;

  outputimage[x + y * outputwidth] = pixelcolor;
}
""").get_function("proc_kernel")

def convolute(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
  image = image.astype(np.uint8)
  kernel = kernel.astype(np.float32)
  kh, kw = kernel.shape
  imh, imw = image.shape[:2]
  newwid = imw - kw
  newhih = imh - kh
  channels = cv2.split(image)
  newchannels = []
  for channel in channels:
    channel = channel.astype(np.uint8)
    newchannel = np.array([[0] * newwid] * newhih, dtype=np.uint8)
    proc_kernel(
      device.In(np.float32(factor)),
      device.In(np.int32(imw)), 
      device.In(np.int32(imh)), 
      device.In(np.int32(kw)), 
      device.In(np.int32(kh)), 
      device.In(channel), 
      device.In(kernel), 
      device.Out(newchannel),
      block=(1024,1,1), 
      grid=((newwid * newhih) // 1024 + 1,1))
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
    [-0.8, -0.4, -0.1, -0.4, -0.8]]),
  "vertical_line" : np.array([
    [-1.0, -0.98, -0.93, -0.7, 0.11, 1, 0.11, -0.7, -0.93, -0.98, -1.0],
    [-1.0, -0.98, -0.93, -0.7, 0.11, 1, 0.11, -0.7, -0.93, -0.98, -1.0],
    [-1.0, -0.98, -0.93, -0.7, 0.11, 1, 0.11, -0.7, -0.93, -0.98, -1.0],
    [-1.0, -0.98, -0.93, -0.7, 0.11, 1, 0.11, -0.7, -0.93, -0.98, -1.0],
    [-1.0, -0.98, -0.93, -0.7, 0.11, 1, 0.11, -0.7, -0.93, -0.98, -1.0]])
  }

active_kernel = "vertical_edge"

if len(sys.argv) >= 3:
  active_kernel = sys.argv[2]

if active_kernel not in kernels:
  input(f"{active_kernel} is not an availble kernel")
  sys.exit()

print("Performing CUDA convolution kernel...")
newim = convolute(im, kernels[active_kernel]*0.1)
print("Saving to file")
newname = impath[:1-impath.rfind(".")] + f"_convoluted_{active_kernel}"
if factor != 1:
  newname += "_f" + str(factor)
newname += ".png"
cv2.imwrite("output/" + newname, newim)
print("Finished.")