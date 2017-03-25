#include <iostream>
#include <cmath>

#include "cuda_runtime.h"
#include "curand.h"
#include "curand_kernel.h"

#include "common.h"
#include "geometry.h"
#include "scene.h"

static const unsigned int BLOCK_DIM = 16;
static const unsigned int RESET_DIM = BLOCK_DIM * BLOCK_DIM;

static int render_width, render_height, render_n;
static int render_count;

static float* render_buffer;
static float* dev_render_buffer;

static sphere_t* dev_spheres;
static surface_t* dev_surfaces;

static curandState* dev_curand_state;

__device__ int kernel_render_width, kernel_render_height;
__device__ float* kernel_render_buffer;
__device__ int kernel_num_spheres;
__device__ sphere_t* kernel_spheres;
__device__ surface_t* kernel_surfaces;
__device__ curandState* kernel_curand_state;

__device__ bool sphere_t::intersect(vec3 start, vec3 direction, float& t, vec3& normal) {

	float a = direction.magnitude_2();
	vec3 recentered = start - center;
	float b = 2 * direction.dot(recentered);
	float c = recentered.magnitude_2() - (radius * radius);

	float discrim = (b * b) - (4.0f * a * c);
	if (discrim < 0.0f) return false;

	float sqrt_discrim = std::sqrtf(discrim);
	float t1 = (-b + sqrt_discrim) / (2.0f * a);
	float t2 = (-b - sqrt_discrim) / (2.0f * a);

	if (t1 > 0.0f && t2 > 0.0f) {
		t = std::fminf(t1, t2);
	} else {
		t = std::fmaxf(t1, t2);
	}
	if (t < 0.0f) return false;

	vec3 surface_point = start + direction * t;
	normal = (surface_point - center) / radius;

	return true;

}

__global__ void resetRenderKernel(float* render_buffer, curandState* curand_state, sphere_t* spheres, surface_t* surfaces, int render_width, int render_height, int num_spheres) {

	int t = blockDim.x * blockIdx.x + threadIdx.x;

	if (t == 0) {
		kernel_render_width = render_width;
		kernel_render_height = render_height;
		kernel_render_buffer = render_buffer;
		kernel_num_spheres = num_spheres;
		kernel_spheres = spheres;
		kernel_surfaces = surfaces;
		kernel_curand_state = curand_state;
	}

	if (t < render_width * render_height) {
		curand_init(t, 0, 0, &curand_state[t]);
	}

}

__global__ void renderKernel(camera_t camera) {

	int x = BLOCK_DIM * blockIdx.x + threadIdx.x;
	int y = BLOCK_DIM * blockIdx.y + threadIdx.y;

	if (x < kernel_render_width && y < kernel_render_height) {

		//int n = kernel_render_height * x + y;

		float screen_x = (float) x / kernel_render_width - 0.5f;
		float screen_y = (float) y / kernel_render_height - 0.5f;
		vec3 ray_direction = camera.forward + (camera.right * camera.aspect_ratio * screen_x + camera.up * screen_y);
		ray_direction.normalize();

		color3 final_color = {0.0f, 0.0f, 0.0f};

		float t;
		vec3 normal;

		float best_t = INFINITY;
		vec3 best_normal;
		int best_surface;

		for (int i = 0; i < kernel_num_spheres; i++) {
			if (kernel_spheres[i].intersect(camera.position, ray_direction, t, normal)) {
				if (t < best_t) {
					best_t = t;
					best_normal = normal;
					best_surface = i;
				}
			}
		}

		final_color += kernel_surfaces[best_surface].emit;

		kernel_render_buffer[kernel_render_width * 3 * y + 3 * x + 0] += final_color.r;
		kernel_render_buffer[kernel_render_width * 3 * y + 3 * x + 1] += final_color.g;
		kernel_render_buffer[kernel_render_width * 3 * y + 3 * x + 2] += final_color.b;

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

bool resetRender(int width, int height, scene_t& scene) {

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
		return false;
	}

	if (cudaMalloc(&dev_spheres, scene.spheres.size() * sizeof(sphere_t)) != cudaSuccess) {
		std::cout << "Cannot allocate enough GPU memory." << std::endl;
		return false;
	}

	if (cudaMemcpy(dev_spheres, scene.spheres.data(), scene.spheres.size() * sizeof(sphere_t), cudaMemcpyHostToDevice) != cudaSuccess) {
		std::cout << "Cannot upload spheres." << std::endl;
		return false;
	}

	if (cudaMalloc(&dev_surfaces, scene.spheres.size() * sizeof(surface_t)) != cudaSuccess) {
		std::cout << "Cannot allocate enough GPU memory." << std::endl;
		return false;
	}

	if (cudaMemcpy(dev_surfaces, scene.surfaces.data(), scene.spheres.size() * sizeof(surface_t), cudaMemcpyHostToDevice) != cudaSuccess) {
		std::cout << "Cannot upload surfaces." << std::endl;
		return false;
	}

	if (cudaMalloc(&dev_curand_state, render_width * render_height * sizeof(curandState)) != cudaSuccess) {
		std::cout << "Cannot allocate enough GPU memory." << std::endl;
		return false;
	}

	int blocks = (render_width * render_height + RESET_DIM - 1) / RESET_DIM;
	int threads_per_block = RESET_DIM;
	resetRenderKernel<<<blocks, threads_per_block>>>(dev_render_buffer, dev_curand_state, dev_spheres, dev_surfaces, render_width, render_height, scene.spheres.size());
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

	return true;

}

bool render(unsigned char* image_data, camera_t& camera) {

	render_count++;

	dim3 blocks((render_width + BLOCK_DIM - 1) / BLOCK_DIM, (render_height + BLOCK_DIM - 1) / BLOCK_DIM);
	dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
	renderKernel<<<blocks, threads_per_block>>>(camera);

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
