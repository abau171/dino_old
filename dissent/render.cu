#include <iostream>
#include <cmath>

#include "cuda_runtime.h"
#include "curand.h"
#include "curand_kernel.h"

static const unsigned int BLOCK_DIM = 16;
static const unsigned int RESET_DIM = BLOCK_DIM * BLOCK_DIM;

static int render_width, render_height, render_n;
static int render_count;
static float* render_buffer;
static float* dev_render_buffer;

static curandState* dev_curand_state;

__global__ void resetRenderKernel(curandState* curand_state, int render_res) {

	int t = blockDim.x * blockIdx.x + threadIdx.x;

	if (t < render_res) {
		curand_init(t, 0, 0, &curand_state[t]);
	}

}

__global__ void renderKernel(float* render_buffer, curandState* curand_state, int render_width, int render_height) {

	int x = BLOCK_DIM * blockIdx.x + threadIdx.x;
	int y = BLOCK_DIM * blockIdx.y + threadIdx.y;

	if (x < render_width && y < render_height) {

		int t = render_height * x + y;

		render_buffer[render_height * 3 * x + 3 * y + 0] += curand_uniform(&curand_state[t]);
		render_buffer[render_height * 3 * x + 3 * y + 1] += curand_uniform(&curand_state[t]);
		render_buffer[render_height * 3 * x + 3 * y + 2] += curand_uniform(&curand_state[t]);

	}

}

bool clearRenderBuffer() {

	if (cudaMemset(dev_render_buffer, 0, render_n * sizeof(float)) != cudaSuccess) {
		std::cout << "Cannot clear render buffer." << std::endl;
		return false;
	}

	return true;

}

bool downloadRenderBuffer() {

	if (cudaMemcpy(render_buffer, dev_render_buffer, render_n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
		std::cout << "Cannot download render buffer." << std::endl;
		return false;
	}

	return true;

}

bool resetRender(int width, int height) {

	render_width = width;
	render_height = height;
	render_n = render_width * render_height * 3;
	render_count = 0;

	if (render_buffer != nullptr) {
		delete render_buffer;
	}
	render_buffer = new float[render_n];

	if (cudaSetDevice(0) != cudaSuccess) {
		std::cout << "Cannot find CUDA device." << std::endl;
		return false;
	}

	if (cudaMalloc(&dev_render_buffer, render_n * sizeof(float)) != cudaSuccess) {
		std::cout << "Cannot allocate enough GPU memory." << std::endl;
		return false;
	}

	if (!clearRenderBuffer()) {
		cudaFree(dev_render_buffer);
		return false;
	}

	if (cudaMalloc(&dev_curand_state, render_width * render_height * sizeof(curandState)) != cudaSuccess) {
		std::cout << "Cannot allocate enough GPU memory." << std::endl;
		cudaFree(dev_render_buffer);
		return false;
	}

	int blocks = (render_width * render_height + RESET_DIM - 1) / RESET_DIM;
	int threads_per_block = RESET_DIM;
	resetRenderKernel<<<blocks, threads_per_block>>>(dev_curand_state, render_width * render_height);
	cudaError_t cudaStatus;

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cout << "Error launching render kernel: " << cudaGetErrorString(cudaStatus) << std::endl;
		cudaFree(dev_render_buffer);
		cudaFree(dev_curand_state);
		return false;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cout << "Error synchronizing with device: " << cudaGetErrorString(cudaStatus) << std::endl;
		cudaFree(dev_render_buffer);
		cudaFree(dev_curand_state);
		return false;
	}

	return true;

}

bool render(unsigned char* image_data) {

	render_count++;

	dim3 blocks((render_width + BLOCK_DIM - 1) / BLOCK_DIM, (render_height + BLOCK_DIM - 1) / BLOCK_DIM);
	dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
	renderKernel<<<blocks, threads_per_block>>>(dev_render_buffer, dev_curand_state, render_width, render_height);

	cudaError_t cudaStatus;

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cout << "Error launching render kernel: " << cudaGetErrorString(cudaStatus) << std::endl;
		return false;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cout << "Error synchronizing with device: " << cudaGetErrorString(cudaStatus) << std::endl;
		return false;
	}

	if (!downloadRenderBuffer()) {
		return false;
	}

	for (int i = 0; i < render_width * render_height * 3; i++) {
		image_data[i] = fminf((256.0f * render_buffer[i]) / render_count, 255.0f);
	}

	return true;

}
