import pycuda.autoinit
import pycuda.driver as device
from pycuda.compiler import SourceModule as cpp

import numpy as np
import sys
import cv2

modify_image = cpp("""
__global__ void modify_image(int* pixelcount, int* width, unsigned char* inputimage, unsigned char* outputimage)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= *pixelcount)
    return;

  outputimage[id] = 255 - inputimage[id];
}
""").get_function("modify_image")

print("Loading image")

image = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED).astype(np.uint8)

print("Processing image")

pixels = image.shape[0] * image.shape[1]
newchannels = []
for channel in cv2.split(image):
  output = np.zeros_like(channel)
  modify_image(
    device.In(np.int32(pixels)),
    device.In(np.int32(image.shape[0])),
    device.In(channel),
    device.Out(output),
    block=(1024,1,1), grid=(pixels // 1024 + 1, 1))
  newchannels.append(output)
finalimage = cv2.merge(newchannels)

print("Saving image")

cv2.imwrite("processed.png", finalimage)

print("Done")