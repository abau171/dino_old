#define _USE_MATH_DEFINES

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
static bool should_clear = false;

static color3* render_buffer;
static color3* dev_render_buffer;

static sphere_t* dev_spheres;
static surface_t* dev_surfaces;

static curandState* dev_curand_state;

__device__ int kernel_render_width, kernel_render_height, kernel_render_n;
__device__ color3* kernel_render_buffer;
__device__ scene_parameters_t kernel_scene_params;
__device__ int kernel_num_spheres;
__device__ sphere_t* kernel_spheres;
__device__ surface_t* kernel_surfaces;
__device__ curandState* kernel_curand_state;

__device__ bool sphere_t::intersect(vec3 start, vec3 direction, float& t, vec3& normal, bool& exiting) {

	float a = direction.magnitude_2();
	vec3 recentered = start - center;
	float b = 2 * direction.dot(recentered);
	float recentered_radius_2 = recentered.magnitude_2();
	float c = recentered_radius_2 - (radius * radius);

	float discrim = (b * b) - (4.0f * a * c);
	if (discrim < 0.0f) return false;

	float sqrt_discrim = std::sqrtf(discrim);
	float t1 = (-b + sqrt_discrim) / (2.0f * a);
	float t2 = (-b - sqrt_discrim) / (2.0f * a);

	exiting = recentered_radius_2 < radius * radius;

	if (exiting) {
		t = std::fmaxf(t1, t2);
	} else {
		t = std::fminf(t1, t2);
	}
	if (t < 0.0f) return false;

	vec3 surface_point = start + direction * t;
	normal = (surface_point - center) / radius;
	if (exiting) normal = -normal;

	return true;

}

__device__ vec3 random_hemi_normal(vec3 normal, int n) {
	vec3 hemi;
	do {
		hemi.x = 2.0f * curand_uniform(&kernel_curand_state[n]) - 1.0f;
		hemi.y = 2.0f * curand_uniform(&kernel_curand_state[n]) - 1.0f;
		hemi.z = 2.0f * curand_uniform(&kernel_curand_state[n]) - 1.0f;
	} while (hemi.magnitude_2() > 1);
	hemi.normalize();
	if (hemi.dot(normal) < 0.0f) {
		hemi = -hemi;
	}
	return hemi;
}

__device__ vec3 confusion_disk(vec3 ortho1, vec3 ortho2, int n) {
	float theta = 2.0f * M_PI * curand_uniform(&kernel_curand_state[n]);
	float sqrtr = sqrtf(curand_uniform(&kernel_curand_state[n]));
	return ortho1 * sqrtr * sinf(theta) + ortho2 * sqrtr * cosf(theta);
}

__global__ void resetRenderKernel(color3* render_buffer, curandState* curand_state, sphere_t* spheres, surface_t* surfaces, int render_width, int render_height, scene_parameters_t scene_params, int num_spheres) {

	int t = blockDim.x * blockIdx.x + threadIdx.x;
	int render_n = render_width * render_height;

	if (t == 0) {
		kernel_render_width = render_width;
		kernel_render_height = render_height;
		kernel_render_n = render_n;
		kernel_render_buffer = render_buffer;
		kernel_scene_params = scene_params;
		kernel_num_spheres = num_spheres;
		kernel_spheres = spheres;
		kernel_surfaces = surfaces;
		kernel_curand_state = curand_state;
	}

	if (t < render_n) {
		curand_init(t, 0, 0, &curand_state[t]);
	}

}

__global__ void renderKernel(camera_t camera) {

	int x = BLOCK_DIM * blockIdx.x + threadIdx.x;
	int y = BLOCK_DIM * blockIdx.y + threadIdx.y;

	if (x < kernel_render_width && y < kernel_render_height) {

		int n = kernel_render_height * x + y;

		float screen_x = (x + curand_uniform(&kernel_curand_state[n]) - 0.5f) / kernel_render_width - 0.5f;
		float screen_y = (y + curand_uniform(&kernel_curand_state[n]) - 0.5f) / kernel_render_height - 0.5f;
		vec3 dof_confusion = confusion_disk(camera.up, camera.right, n) * kernel_scene_params.aperture_radius;
		vec3 ray_start = camera.position + dof_confusion;
		vec3 ray_direction = (camera.forward + camera.right * camera.aspect_ratio * screen_x + camera.up * screen_y) * kernel_scene_params.focal_distance - dof_confusion;
		ray_direction.normalize();

		color3 final_color = {0.0f, 0.0f, 0.0f};
		color3 d_product = {1.0f, 1.0f, 1.0f};

		for (int depth = 0; depth < 15; depth++) {

			float t;
			vec3 normal;
			bool exiting;

			float best_t = INFINITY;
			vec3 best_normal;
			int best_surface;
			bool best_exiting;

			for (int i = 0; i < kernel_num_spheres; i++) {
				if (kernel_spheres[i].intersect(ray_start, ray_direction, t, normal, exiting)) {
					if (t < best_t) {
						best_t = t;
						best_normal = normal;
						best_surface = i;
						best_exiting = exiting;
					}
				}
			}

			if (best_t < INFINITY) {

				ray_start = ray_start + ray_direction * best_t;
				vec3 off_surface = best_normal * 0.0001f;

				float effective_reflectance;
				float effective_transmission;

				float n1 = best_exiting ? kernel_surfaces[best_surface].refractive_index : 1.0f;
				float n2 = best_exiting ? 1.0f : kernel_surfaces[best_surface].refractive_index;
				float ni = n1 / n2;

				float cosi = -ray_direction.dot(best_normal);
				float sint_2 = ni * ni * (1 - cosi * cosi);
				float cost = sqrtf(1 - sint_2);

				if (kernel_surfaces[best_surface].reflectance > 0.0f) {

					if (sint_2 > 1) {
						effective_reflectance = 1.0f;
					} else {
						float r0 = (n1 - n2) / (n1 + n2);
						r0 *= r0;
						float base;
						if (n1 <= n2) {
							base = 1.0f - cosi;
						} else {
							base = 1.0f - cost;
						}
						float r_schlick = r0 + (1 - r0) * base * base * base * base * base;
						effective_reflectance = kernel_surfaces[best_surface].reflectance + (1.0f - kernel_surfaces[best_surface].reflectance) * r_schlick;
					}
				} else {
					effective_reflectance = 0.0f;
				}

				if (curand_uniform(&kernel_curand_state[n]) < effective_reflectance) {

					ray_start += off_surface;
					ray_direction = ray_direction.reflect(best_normal);

					final_color += d_product * kernel_surfaces[best_surface].emit;
					d_product *= kernel_surfaces[best_surface].specular;

				} else if (curand_uniform(&kernel_curand_state[n]) < kernel_surfaces[best_surface].transmission) {

					ray_start -= off_surface;
					ray_direction = ray_direction * ni + best_normal * (ni * cosi - cost);
					ray_direction.normalize();
					if (best_exiting) {
						color3 attenuation_color = kernel_surfaces[best_surface].attenuation_color;
						color3 beer = {expf(best_t * logf(attenuation_color.r)), expf(best_t * logf(attenuation_color.g)), expf(best_t * logf(attenuation_color.b))};
						d_product *= beer;
					}

				} else {

					ray_start += off_surface;
					ray_direction = random_hemi_normal(best_normal, n);

					final_color += d_product * kernel_surfaces[best_surface].emit;
					d_product *= kernel_surfaces[best_surface].diffuse;

				}
			} else {

				final_color += d_product * kernel_scene_params.background_emission;
				break;
			}

		}

		kernel_render_buffer[kernel_render_width * y + x] += final_color;

	}

}

bool clearRenderBuffer() {

	if (cudaMemset(dev_render_buffer, 0, render_n * sizeof(color3)) != cudaSuccess) {
		std::cout << "Cannot clear render buffer." << std::endl;
		return false;
	}

	return true;

}

bool downloadRenderBuffer() {

	if (cudaMemcpy(render_buffer, dev_render_buffer, render_n * sizeof(color3), cudaMemcpyDeviceToHost) != cudaSuccess) {
		std::cout << "Cannot download render buffer." << std::endl;
		return false;
	}

	return true;

}

bool resetRender(int width, int height, scene_t& scene) {

	render_width = width;
	render_height = height;
	render_n = render_width * render_height;
	render_count = 0;

	if (render_buffer != nullptr) {
		delete render_buffer;
	}
	render_buffer = new color3[render_n];

	if (cudaSetDevice(0) != cudaSuccess) {
		std::cout << "Cannot find CUDA device." << std::endl;
		return false;
	}

	if (cudaMalloc(&dev_render_buffer, render_n * sizeof(color3)) != cudaSuccess) {
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

	if (cudaMalloc(&dev_curand_state, render_n * sizeof(curandState)) != cudaSuccess) {
		std::cout << "Cannot allocate enough GPU memory." << std::endl;
		return false;
	}

	int blocks = (render_n + RESET_DIM - 1) / RESET_DIM;
	int threads_per_block = RESET_DIM;
	resetRenderKernel<<<blocks, threads_per_block>>>(dev_render_buffer, dev_curand_state, dev_spheres, dev_surfaces, render_width, render_height, scene.params, scene.spheres.size());
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

	if (should_clear) {
		clearRenderBuffer();
		render_count = 0;
		should_clear = false;
	}

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

	for (int i = 0; i < render_n; i++) {
		color3 color = render_buffer[i] * (256.0f / render_count);
		image_data[3 * i] = fminf(color.r, 255.0f);
		image_data[3 * i + 1] = fminf(color.g, 255.0f);
		image_data[3 * i + 2] = fminf(color.b, 255.0f);
	}

	std::cout << render_count << std::endl;

	return true;

}

void clearRender() {

	render_count = 0;
	should_clear = true;

}
