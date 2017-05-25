import pycuda.autoinit
import pycuda.driver as device
from pycuda.compiler import SourceModule as source

import numpy as np
import sys, os
import cv2
import subprocess

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

factor = 16.0

if len(sys.argv) >= 3:
  factor = float(sys.argv[2])

frames = 60

if len(sys.argv) >= 4:
  frames = int(sys.argv[3])

pass_filter = source("""
__global__ void rapidsine_filter(int* width, int* height, float* offset, unsigned char* image, unsigned char* output)
{
  const int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= (*width) * (*height))
    return;

  int x = id % *height;
  int y = id / *height;

  int newvalue = image[x + y * (*height)];
  newvalue = 127.0f * sinf((3.14159265f * (newvalue + *offset)) / """ + str(factor) + """f) + 127.0f;
  output[x + y * (*height)] = newvalue;
}
""").get_function("rapidsine_filter")

outputfolder = impath[:impath.rfind(".")] + "_filtered_output"
try: os.makedirs(outputfolder)
except: pass

print("Creating frames")

for i in range(frames):
  offset = (factor * 2) * (i / frames)
  new_channels = []
  for channel in cv2.split(image):
    width, height = channel.shape
    output = np.zeros_like(channel)
    pass_filter(
      device.In(np.int32(width)),
      device.In(np.int32(height)), 
      device.In(np.float32(offset)),
      device.In(channel), 
      device.Out(output),
      block=(1024, 1, 1), 
      grid=((width * height) // 1024 + 1, 1))
    new_channels.append(output)
  finalimage = cv2.merge(new_channels)
  print(f"Saving frame {i+1} of {frames}")
  cv2.imwrite(outputfolder + f"/frame_{i}.png", finalimage)

print("Combining into video")

os.chdir(outputfolder)
subprocess.call([
  "ffmpeg",
  "-framerate", "60",
  "-i", "frame_%d.png",
  "-crf", "12",
  "../" + outputfolder + ".mp4"])

print("Done")