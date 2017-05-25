import pycuda.autoinit
import pycuda.driver as device
from pycuda.compiler import SourceModule as source

import numpy as np
import sys, os
import cv2

active_filter = "square_filter"

print("Loading image")

if len(sys.argv) < 2:
  print("Please give image file argument")
  sys.exit()

impath = sys.argv[1]

if not os.path.isfile(impath):
  print("File not exist")
  sys.exit()

image = cv2.imread(impath, cv2.IMREAD_UNCHANGED).astype(np.uint8)

if image is None:
  print("Invalid image file")
  sys.exit()

if len(sys.argv) >= 3:
  active_filter = sys.argv[2]

pass_filter = source("""
__global__ void blur_filter(int* width, int* height, unsigned char* image, unsigned char* output)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= (*width) * (*height))
    return;

  int x = id % *height;
  int y = id / *height;

  int newvalue = 0;
  int numvals = 0;
  for (int yd = -20; yd <= 20; yd++)
  {
    for (int xd = -20; xd <= 20; xd++)
    {
      int nx = x + xd;
      int ny = y + yd;
      if (nx < 0 || nx >= *height || ny < 0 || ny >= *width)
        continue;
      numvals++;
      newvalue += image[nx + ny * (*height)];
    }
  }
  newvalue = newvalue / numvals;
  output[x + y * (*height)] = newvalue;
}

__global__ void step_filter(int* width, int* height, unsigned char* image, unsigned char* output)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= (*width) * (*height))
    return;

  int x = id % *height;
  int y = id / *height;

  unsigned char newvalue = image[x + y * (*height)];
  newvalue = newvalue - (newvalue % 128);
  output[x + y * (*height)] = newvalue;
}

__global__ void logstep_filter(int* width, int* height, unsigned char* image, unsigned char* output)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= (*width) * (*height))
    return;

  int x = id % *height;
  int y = id / *height;

  int newvalue = image[x + y * (*height)];
  newvalue = (unsigned char)powf(2.0f, floorf(log2f((float)newvalue)));
  output[x + y * (*height)] = newvalue;
}

__global__ void square_filter(int* width, int* height, unsigned char* image, unsigned char* output)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= (*width) * (*height))
    return;

  int x = id % *height;
  int y = id / *height;

  int newvalue = image[x + y * (*height)];
  newvalue = powf((newvalue + 1) / 256.0f, 2.0f) * 256.0f - 1.0f;
  output[x + y * (*height)] = newvalue;
}

__global__ void root_filter(int* width, int* height, unsigned char* image, unsigned char* output)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= (*width) * (*height))
    return;

  int x = id % *height;
  int y = id / *height;

  int newvalue = image[x + y * (*height)];
  newvalue = sqrtf((newvalue + 1) / 256.0f) * 256.0f - 1.0f;
  output[x + y * (*height)] = newvalue;
}

__global__ void sine_filter(int* width, int* height, unsigned char* image, unsigned char* output)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= (*width) * (*height))
    return;

  int x = id % *height;
  int y = id / *height;

  int newvalue = image[x + y * (*height)];
  newvalue = 255.0f * sinf((3.14159265f * newvalue) / 255.0f);
  output[x + y * (*height)] = newvalue;
}

__global__ void twosine_filter(int* width, int* height, unsigned char* image, unsigned char* output)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= (*width) * (*height))
    return;

  int x = id % *height;
  int y = id / *height;

  int newvalue = image[x + y * (*height)];
  newvalue = 127.0f * sinf((3.14159265f * newvalue) / 127.0f) + 127.0f;
  output[x + y * (*height)] = newvalue;
}

__global__ void rapidsine_filter(int* width, int* height, unsigned char* image, unsigned char* output)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= (*width) * (*height))
    return;

  int x = id % *height;
  int y = id / *height;

  int newvalue = image[x + y * (*height)];
  newvalue = 127.0f * sinf((3.14159265f * newvalue) / 16.0f) + 127.0f;
  output[x + y * (*height)] = newvalue;
}

""").get_function(active_filter)

print("Processing image")

new_channels = []
for channel in cv2.split(image):
  width, height = channel.shape
  output = np.zeros_like(channel)
  pass_filter(
    device.In(np.int32(width)), device.In(np.int32(height)),
    device.In(channel), device.Out(output),
    block=(1024, 1, 1), grid=((width * height) // 1024 + 1, 1))
  # print(output)
  new_channels.append(output)
finalimage = cv2.merge(new_channels)

print("Saving image")

try: os.makedirs("output")
except: pass
cv2.imwrite("output/" + impath[:impath.rfind(".")] + f"_filtered_{active_filter}.png", finalimage)

print("Done")