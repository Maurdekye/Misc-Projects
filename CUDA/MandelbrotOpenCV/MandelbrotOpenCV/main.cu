
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <windows.h>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv\cv.h>

using namespace std;

#define resX 1920
#define resY 1080
#define threadcount 1024

double targetX = -1.749957681f;
double targetY = 0.0f;

int numiters = 100;

struct dimensions {
	double lowX;
	double highX;
	double lowY;
	double highY;
};

dimensions gridDimensions = { -2.0f, 1.2f, -0.9f, 0.9f };

unsigned long long int getCurTime()
{
	return chrono::duration_cast<chrono::milliseconds>(chrono::system_clock().now().time_since_epoch()).count();
}

int SolveMandelbrot(double real, double imag, int maxiters)
{
	double oreal, oimag;
	oreal = real;
	oimag = imag;
	for (; maxiters > 0; maxiters--)
	{
		double newreal = (real * real - imag * imag) + oreal;
		double newimag = (2 * real * imag) + oimag;
		real = newreal;
		imag = newimag;
		if (real * real + imag * imag > 4)
			break;
	}
	return maxiters;
}

void MandelbrotCPU(int** numgrid, int iters)
{
	double xRange = gridDimensions.highX - gridDimensions.lowX;
	double yRange = gridDimensions.highY - gridDimensions.lowY;
	for (int y = 0; y < resY; y++)
	{
		for (int x = 0; x < resX; x++)
		{
			double xPercent = static_cast<double>(x) / resX;
			double yPercent = static_cast<double>(y) / resY;
			double xPos = xPercent * xRange + gridDimensions.lowX;
			double yPos = yPercent * yRange + gridDimensions.lowY;
			int calculation = SolveMandelbrot(xPos, yPos, iters);
			numgrid[x][y] = calculation;
		}
	}
}

__global__ void SolveMandelbrotKernel(int* numgrid, int iters, int gridSize, dimensions gridDimensions)
{
	int idx = blockIdx.x * threadcount + threadIdx.x;
	if (idx >= gridSize)
		return;

	int x = idx % resX;
	int y = idx / resX;
	double xPercent = static_cast<double>(x) / resX;
	double yPercent = static_cast<double>(y) / resY;
	double xRange = gridDimensions.highX - gridDimensions.lowX;
	double yRange = gridDimensions.highY - gridDimensions.lowY;
	double real = xPercent * xRange + gridDimensions.lowX;
	double imag = yPercent * yRange + gridDimensions.lowY;

	double oreal = real;
	double oimag = imag;
	int maxiters = iters;
	for (; maxiters > 0; maxiters--)
	{
		double newreal = (real * real - imag * imag) + oreal;
		double newimag = (2 * real * imag) + oimag;
		real = newreal;
		imag = newimag;
		if (real * real + imag * imag > 4)
			break;
	}

	numgrid[idx] = maxiters;
}

cudaError_t MandelbrotCUDA(int** numgrid, int iters)
{
	cudaError_t cudaErr = cudaSuccess;

	unsigned long long init = getCurTime();
	cudaErr = cudaSetDevice(0);
	if (cudaErr != cudaSuccess)
	{
		cout << "Device set failure!" << endl;
		return cudaErr;
	}

	int ngSize = resX * resY * sizeof(int);
	int* dev_ng;

	unsigned long long start = getCurTime();
	cudaErr = cudaMalloc(&dev_ng, ngSize);
	if (cudaErr != cudaSuccess)
	{
		cout << "Device memory allocation failure!" << endl;
		return cudaErr;
	}
	unsigned long long pre = getCurTime();
	SolveMandelbrotKernel << <ngSize / threadcount + 1, threadcount >> >(dev_ng, iters, resX * resY, gridDimensions);
	cudaErr = cudaDeviceSynchronize();
	if (cudaErr != cudaSuccess)
	{
		cout << "Device kernel execution failure!" << endl;
		return cudaErr;
	}
	unsigned long long post = getCurTime();
	int* flatNumGrid = (int*)malloc(ngSize);
	cudaErr = cudaMemcpy(flatNumGrid, dev_ng, ngSize, cudaMemcpyDeviceToHost);
	if (cudaErr != cudaSuccess)
	{
		cout << "Device memory retention failure!" << endl;
		return cudaErr;
	}

	for (int y = 0; y < resY; y++)
	{
		for (int x = 0; x < resX; x++)
		{
			numgrid[x][y] = flatNumGrid[y * resX + x];
		}
	}

	cudaFree(dev_ng);
	delete flatNumGrid;
	unsigned long long cleanup = getCurTime();
	cudaDeviceReset();

	cout << "Initializations: " << (start - init) << " milliseconds" << endl;
	cout << "Initial memory allocation: " << (pre - start) << " milliseconds" << endl;
	cout << "Processing: " << (post - pre) << " milliseconds" << endl;
	cout << "Memory Cleanup: " << (cleanup - post) << " milliseconds" << endl;
	cout << endl << "Total start to end: " << (cleanup - init) << " milliseconds" << endl;

	return cudaErr;
}

void PrintNumgrid(int** numgrid)
{
	for (int y = 0; y < resY; y++)
	{
		for (int x = 0; x < resX; x++)
		{
			if (numgrid[x][y] == 0)
			{
				cout << 0;
			}
			else {
				cout << " ";
				;
			}
			cout << " ";
		}
		cout << "|" << endl;
	}
	for (int i = 0; i < resX * 2; i++)
		cout << "-";
	cout << "+" << endl;
}

void SaveToIm(int** numgrid, string filename)
{
	cv::Mat* im = new cv::Mat(resY, resX, CV_8UC3, cv::Scalar(0, 0, 0));

	for (int y = 0; y < resY; y++)
	{
		for (int x = 0; x < resX; x++)
		{
			double percent = numgrid[x][y] / (numiters * 1.0f);
			//double fval = numiters - 32.0f * log2f(numiters - numgrid[x][y]);
			int val = pow(percent, 2) * 255;

			cv::Vec3b pixel = im->at<cv::Vec3b>(cv::Point(x, y));
			pixel[0] = val;
			pixel[1] = val;
			pixel[2] = val;
			im->at<cv::Vec3b>(cv::Point(x, y)) = pixel;
		}
	}

	cv::imwrite(filename, *im);
	delete im;
}

void ZoomToTarget(double factor)
{
	gridDimensions.lowX -= targetX;
	gridDimensions.highX -= targetX;
	gridDimensions.lowY -= targetY;
	gridDimensions.highY -= targetY;

	gridDimensions.lowX *= factor;
	gridDimensions.highX *= factor;
	gridDimensions.lowY *= factor;
	gridDimensions.highY *= factor;

	gridDimensions.lowX += targetX;
	gridDimensions.highX += targetX;
	gridDimensions.lowY += targetY;
	gridDimensions.highY += targetY;
}

int main()
{
	string outputDir = "output";
	CreateDirectory(outputDir.c_str(), NULL);

	int** numbers = new int*[resX];
	for (int a = 0; a < resX; a++)
		numbers[a] = new int[resY];

	int frames = 1800;

	for (int i = 0; i < frames; i++)
	{
		unsigned long long start = getCurTime();

		//MandelbrotCPU(numbers, numiters);
		MandelbrotCUDA(numbers, numiters);

		unsigned long long end = getCurTime();

		cout << "Time: " << (end - start) << " milliseconds" << endl;
		cout << "Saving to image file..." << endl;

		string imageLocation = outputDir + "\\" + to_string(i) + ".png";
		SaveToIm(numbers, imageLocation);
		cout << "Saved frame " << i+1 << "/" << frames << endl;

		ZoomToTarget(0.99f);
		numiters += 20;
	}

	for (int a = 0; a < resX; a++)
		delete [] numbers[a];
	delete [] numbers;

	cout << "Finished all frames." << endl;
	cin.ignore();
	return 0;
}
