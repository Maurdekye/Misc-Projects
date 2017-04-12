#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include <GLFW/glfw3.h>

using namespace std;

int main()
{
	cout << "initializing" << endl;

	// Initialize GLFW
	if (!glfwInit())
	{
		cerr << "Failed to initialize GLFW" << endl;
		return 1;
	}

	cout << "setting flags" << endl;

	//glfwWindowHint(GLFW_SAMPLES, 4);

	cout << "creating window" << endl;

	// create display window
	GLFWwindow* frame = glfwCreateWindow(1024, 768, "Mandelbrot", NULL, NULL);
	if (frame == NULL)
	{
		cerr << "Failed to open window" << endl;
		glfwTerminate();
		return 2;
	}

	cout << "initializing context" << endl;

	glfwMakeContextCurrent(frame);

	// wait for keypress
	glfwSetInputMode(frame, GLFW_STICKY_KEYS, GL_TRUE);

	cout << "waiting for keypress" << endl;

	do {
		glfwSwapBuffers(frame);
		glfwPollEvents();
	} while (glfwGetKey(frame, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(frame) != 0);

	cout << "press any key to exit..." << endl;
	cin.ignore();

	glfwTerminate();
	return 0;
}