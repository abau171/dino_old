#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
#include "GL/glew.h"
#include "GL/glut.h"
#include "cuda_runtime.h"
#include "curand.h"
#include "curand_kernel.h"
#include "cuda_gl_interop.h"

#include "common.h"
#include "geometry.h"
#include "scene.h"

#include "render.h"

#define MAX_DEPTH 4

static const unsigned int BLOCK_DIM = 16;

static int render_width, render_height, render_n;
static int render_count;
static bool should_clear = false;

static color3* dev_render_buffer;
static float* dev_output_buffer;

static GLuint gl_image_buffer;

static curandState* dev_curand_state;
static sphere_t* dev_spheres;
static material_t* dev_materials;

__device__ int kernel_render_width, kernel_render_height, kernel_render_n;
__device__ color3* kernel_render_buffer;
__device__ scene_parameters_t kernel_scene_params;
__device__ int kernel_num_spheres;
__device__ sphere_t* kernel_spheres;
__device__ material_t* kernel_materials;
__device__ curandState* kernel_curand_state;

__device__ float sphere_t::intersect(vec3 start, vec3 direction) {

	float a = direction.magnitude_2();
	vec3 recentered = start - center;
	float b = 2 * direction.dot(recentered);
	float recentered_radius_2 = recentered.magnitude_2();
	float c = recentered_radius_2 - (radius * radius);

	float discrim = (b * b) - (4.0f * a * c);
	if (discrim < 0.0f) return -1.0f;

	float sqrt_discrim = std::sqrtf(discrim);
	float t1 = (-b + sqrt_discrim) / (2.0f * a);
	float t2 = (-b - sqrt_discrim) / (2.0f * a);

	float t;
	if (c < 0.0f) {
		t = std::fmaxf(t1, t2);
	} else {
		t = std::fminf(t1, t2);
	}
	return t;

}

__device__ vec3 random_isotropic(float cos_theta, int n) {

	float phi = 2.0f * M_PI * curand_uniform(&kernel_curand_state[n]);
	float cos_phi = cosf(phi);
	float sin_phi = sinf(phi);

	float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

	return {
		sin_theta * cos_phi,
		cos_theta,
		sin_theta * sin_phi
	};
}

__device__ vec3 random_sphere(int n) {

	float cos_theta = 2.0f * curand_uniform(&kernel_curand_state[n]) - 1.0f;
	return random_isotropic(cos_theta, n);

}

__device__ vec3 random_hemi(int n) {

	float cos_theta = curand_uniform(&kernel_curand_state[n]);
	return random_isotropic(cos_theta, n);

}

__device__ vec3 random_phong_hemi(float spec_power, int n) {

	float cos_theta = powf(curand_uniform(&kernel_curand_state[n]), 1.0f / (spec_power + 1.0f));
	return random_isotropic(cos_theta, n);

}

__device__ vec3 random_henyey_greenstein(float g, int n) {

	float s = 2.0f * curand_uniform(&kernel_curand_state[n]) - 1.0f;

	float cos_theta;
	if (g == 0.0f) {
		cos_theta = s;
	} else {
		float g_2 = g * g;
		float a = (1.0f - g_2) / (1.0f + g * s);
		cos_theta = (1.0f + g_2 - a * a) / (2.0f * g);
	}

	return random_isotropic(cos_theta, n);

}

__device__ vec3 confusion_disk(vec3 ortho1, vec3 ortho2, int n) {
	float theta = 2.0f * M_PI * curand_uniform(&kernel_curand_state[n]);
	float sqrtr = sqrtf(curand_uniform(&kernel_curand_state[n]));
	return ortho1 * sqrtr * sinf(theta) + ortho2 * sqrtr * cosf(theta);
}

__global__ void initRenderKernel(float* output_buffer, color3* render_buffer, curandState* curand_state, sphere_t* spheres, material_t* surfaces, int render_width, int render_height, scene_parameters_t scene_params, int num_spheres) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < render_width && y < render_height) {

		int n = render_height * x + y;

		if (n == 0) {
			kernel_render_width = render_width;
			kernel_render_height = render_height;
			kernel_render_n = render_width * render_height;
			kernel_render_buffer = render_buffer;
			kernel_scene_params = scene_params;
			kernel_num_spheres = num_spheres;
			kernel_spheres = spheres;
			kernel_materials = surfaces;
			kernel_curand_state = curand_state;
		}

		curand_init(n, 0, 0, &curand_state[n]);

		output_buffer[render_width * render_height + 2 * n + 0] = x;
		output_buffer[render_width * render_height + 2 * n + 1] = y;

	}

}

__global__ void renderKernel(output_color_t* output_buffer, camera_t camera, int render_count) {

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
		color3 running_absorption = {1.0f, 1.0f, 1.0f};

		volume_t cur_volume = kernel_scene_params.air_volume;

		for (int depth = 0; depth < MAX_DEPTH; depth++) {

			float t = INFINITY;
			int sphere_index = -1;

			for (int i = 0; i < kernel_num_spheres; i++) {
				float test_t = kernel_spheres[i].intersect(ray_start, ray_direction);
				if (test_t >= 0.0f) {
					if (test_t < t) {
						t = test_t;
						sphere_index = i;
					}
				}
			}

			float scatter_t = (cur_volume.scatter > 0.0f) ? -logf(curand_uniform(&kernel_curand_state[n])) / cur_volume.scatter : INFINITY;

			if (t > scatter_t) { // scatter

				color3 attenuation = cur_volume.attenuation;
				color3 beer = { // shortcut if any component is zero to get rid of fireflies
					attenuation.r > 0.0f ? expf(scatter_t * logf(attenuation.r)) : 0.0f,
					attenuation.g > 0.0f ? expf(scatter_t * logf(attenuation.g)) : 0.0f,
					attenuation.b > 0.0f ? expf(scatter_t * logf(attenuation.b)) : 0.0f
				};
				running_absorption *= beer;

				ray_start += ray_direction * scatter_t;
				ray_direction = random_henyey_greenstein(cur_volume.scatter_g, n).change_up(ray_direction);

			} else if (sphere_index != -1) { // interact with surface

				vec3 surface_position = ray_start + ray_direction * t;
				vec3 normal = (surface_position - kernel_spheres[sphere_index].center);
				normal.normalize();
				bool exiting = ray_direction.dot(normal) > 0.0f;
				if (exiting) normal = -normal;

				color3 attenuation = cur_volume.attenuation;
				color3 beer = { // shortcut if any component is zero to get rid of fireflies
					attenuation.r > 0.0f ? expf(t * logf(attenuation.r)) : 0.0f,
					attenuation.g > 0.0f ? expf(t * logf(attenuation.g)) : 0.0f,
					attenuation.b > 0.0f ? expf(t * logf(attenuation.b)) : 0.0f
				};
				running_absorption *= beer;

				material_t material = kernel_materials[sphere_index];

				ray_start += ray_direction * t;
				vec3 off_surface = normal * 0.0001f; // add small amount to get off the surface (no shading acne)

				float effective_specular_weight;

				float n1 = exiting ? material.volume.refractive_index : kernel_scene_params.air_volume.refractive_index;
				float n2 = exiting ? kernel_scene_params.air_volume.refractive_index : material.volume.refractive_index;
				float ni = n1 / n2;

				float cosi = -ray_direction.dot(normal);
				float sint_2 = ni * ni * (1 - cosi * cosi);
				float cost = sqrtf(1 - sint_2);

				if (material.surface.specular_weight > 0.0f) {

					if (sint_2 > 1) {
						effective_specular_weight = 1.0f;
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
						effective_specular_weight = material.surface.specular_weight + (1.0f - material.surface.specular_weight) * r_schlick;
					}
				} else {
					effective_specular_weight = 0.0f;
				}

				if (curand_uniform(&kernel_curand_state[n]) < effective_specular_weight) { // specular

					ray_start += off_surface;

					if (material.surface.spec_power > 0.0f) { // Phong specular

						vec3 ray_reflect = ray_direction.reflect(normal);
						ray_direction = random_phong_hemi(material.surface.spec_power, n).change_up(ray_reflect);

					} else { // perfect reflection

						ray_direction = ray_direction.reflect(normal);

					}

					cur_volume = exiting ? material.volume : kernel_scene_params.air_volume;

					final_color += running_absorption * material.surface.emission;
					running_absorption *= material.surface.specular;

				} else if (curand_uniform(&kernel_curand_state[n]) < material.surface.transmission_weight) { // refract

					ray_start -= off_surface;
					ray_direction = ray_direction * ni + normal * (ni * cosi - cost);
					ray_direction.normalize();
					cur_volume = exiting ? kernel_scene_params.air_volume : material.volume;

				} else { // diffuse

					ray_start += off_surface;
					ray_direction = random_phong_hemi(1.0f, n).change_up(normal);
					cur_volume = exiting ? material.volume : kernel_scene_params.air_volume;

					final_color += running_absorption * material.surface.emission;
					running_absorption *= material.surface.diffuse;

				}

			} else {

				final_color += running_absorption * kernel_scene_params.background_emission;
				break;
			}

		}

		final_color += kernel_render_buffer[n];
		kernel_render_buffer[n] = final_color;

		color3 output_color = (final_color / (render_count + 1)).linearToGamma() * 255.0f;
		output_color = {
			fminf(255.0f, output_color.r),
			fminf(255.0f, output_color.g),
			fminf(255.0f, output_color.b),
		};

		output_buffer[n] = {
			(unsigned char) output_color.r,
			(unsigned char) output_color.g,
			(unsigned char) output_color.b,
			255
		};

	}

}

bool clearRenderBuffer() {

	if (cudaMemset(dev_render_buffer, 0, render_n * sizeof(color3)) != cudaSuccess) {
		std::cout << "Cannot clear render buffer." << std::endl;
		return false;
	}

	return true;

}

bool initRender(int width, int height, scene_t& scene, GLuint new_gl_image_buffer) {

	render_width = width;
	render_height = height;
	render_n = render_width * render_height;
	render_count = 0;
	gl_image_buffer = new_gl_image_buffer;

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

	if (cudaMalloc(&dev_materials, scene.spheres.size() * sizeof(material_t)) != cudaSuccess) {
		std::cout << "Cannot allocate enough GPU memory." << std::endl;
		return false;
	}

	if (cudaMemcpy(dev_materials, scene.materials.data(), scene.spheres.size() * sizeof(material_t), cudaMemcpyHostToDevice) != cudaSuccess) {
		std::cout << "Cannot upload materials." << std::endl;
		return false;
	}

	if (cudaMalloc(&dev_curand_state, render_n * sizeof(curandState)) != cudaSuccess) {
		std::cout << "Cannot allocate enough GPU memory." << std::endl;
		return false;
	}

	cudaError_t cudaStatus;

	cudaStatus = cudaGLRegisterBufferObject(gl_image_buffer);
	if (cudaStatus != cudaSuccess) {
		std::cout << "Error registering OpenGL buffer: " << cudaGetErrorString(cudaStatus) << std::endl;
		return false;
	}

	cudaGLMapBufferObject((void**) &dev_output_buffer, gl_image_buffer);

	dim3 blocks((render_width + BLOCK_DIM - 1) / BLOCK_DIM, (render_height + BLOCK_DIM - 1) / BLOCK_DIM);
	dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
	initRenderKernel<<<blocks, threads_per_block>>>(dev_output_buffer, dev_render_buffer, dev_curand_state, dev_spheres, dev_materials, render_width, render_height, scene.params, scene.spheres.size());

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

	cudaGLUnmapBufferObject(gl_image_buffer);

	return true;

}

bool render(camera_t& camera) {

	if (should_clear) {
		clearRenderBuffer();
		render_count = 0;
		should_clear = false;
	}

	render_count++;

	cudaGLMapBufferObject((void**) &dev_output_buffer, gl_image_buffer);

	dim3 blocks((render_width + BLOCK_DIM - 1) / BLOCK_DIM, (render_height + BLOCK_DIM - 1) / BLOCK_DIM);
	dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
	renderKernel<<<blocks, threads_per_block>>>((output_color_t*) dev_output_buffer, camera, render_count);

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

	cudaGLUnmapBufferObject(gl_image_buffer);

	std::cout << render_count << std::endl;

	return true;

}

void clearRender() {

	should_clear = true;

}

output_color_t* downloadOutputBuffer() {

	output_color_t* output_buffer = new output_color_t[render_n];

	cudaGLMapBufferObject((void**) &dev_output_buffer, gl_image_buffer);
	cudaMemcpy(output_buffer, dev_output_buffer, render_n * sizeof(output_color_t), cudaMemcpyDeviceToHost);
	cudaGLUnmapBufferObject(gl_image_buffer);

	return output_buffer;

}
