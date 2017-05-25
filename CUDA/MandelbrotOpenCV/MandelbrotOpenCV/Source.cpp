
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>

__global__ void proc_kernel(int imwidth, int imheight, int kernelwidth, int kernelheight, int* inputimage, float* kernel, int* outputimage)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	int outputwidth = imwidth - kernelwidth;
	int outputheight = imheight - kernelheight;

	if (id >= outputwidth * outputheight)
		return;

	int x = id / outputwidth;
	int y = id % outputwidth;

	int halfkernelwidth = kernelwidth / 2;
	int halfkernelheight = kernelheight / 2;

	float sum = 0.0f;

	for (int kx = 0; kx < kernelwidth; kx++)
	{
		for (int ky = 0; ky < kernelheight; ky++)
		{
			int xd = kx - halfkernelwidth;
			int yd = ky - halfkernelheight;

			sum += kernel[kx * kernelwidth + ky] * inputimage[(x + xd) * imwidth + (y + yd)];
		}
	}

	int sigmoid;
	if (abs(sum) > 100)
		sigmoid = sum / abs(sum);
	else
		sigmoid = 1.0f / 1.0f + exp(-sum);

	int pixelcolor = (sigmoid + 1) * 127;

	outputimage[x * outputwidth + y] = pixelcolor;
}