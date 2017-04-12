#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
using namespace std;

__global__ void check_prime(int offset, int* outputs)
{
	int num = blockIdx.x + offset;
	outputs[num] = 1;
	if (num < 2)
	{
		outputs[num] = 0;
		return;
	}
	for (int i = 2; i < num; i++)
	{
		if (num % i == 0)
		{
			outputs[num] = 0;
			break;
		}
	}
}

int main()
{
	
	cudaSetDevice(0);
	const int arrsize = 65535;
	const int memsize = arrsize * sizeof(int);
	int offset = 2;
	int* dev_ptr;
	int* output[arrsize];
	cudaMalloc(&dev_ptr, memsize);

	while (true)
	{
		//cout << endl << "iteration " << j << endl;
		check_prime << <arrsize, 1 >> >(offset, dev_ptr);
		cudaDeviceSynchronize();
		cudaMemcpy(output, dev_ptr, memsize, cudaMemcpyDeviceToHost);

		for (int i = 0; i < arrsize; i++)
		{
			if (output[i])
				cout << i + offset << " ";
		}

		offset += arrsize;
	}

	cudaFree(dev_ptr);
	cout << endl << "end" << endl;
	cin.ignore();

	return 0;
}