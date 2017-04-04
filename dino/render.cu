#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include "GL/glew.h"
#include "GL/glut.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"
#include "cuda_gl_interop.h"

#include "common.h"
#include "geometry.h"
#include "scene.h"

#include "render.h"

#define MAX_DEPTH 4

#define SPHERE_TYPE 0
#define INSTANCE_TYPE 1

static const unsigned int BLOCK_DIM = 16;

static int render_width, render_height, render_n;
static int render_count;
static bool should_clear = false;
static std::chrono::time_point<std::chrono::steady_clock> start_time;

static color3* dev_render_buffer;
static float* dev_output_buffer;

static GLuint gl_image_buffer;

static curandState* dev_curand_state;
static sphere_instance_t* dev_spheres;
static triangle_t* dev_triangles;
static triangle_extra_t* dev_extras;
static model_t* dev_models;
static std::vector<bvh_node_t*> dev_bvhs;
static std::vector<color3*> dev_textures;
static instance_t* dev_instances;

__device__ int kernel_render_width, kernel_render_height, kernel_render_n;
__device__ color3* kernel_render_buffer;
__device__ scene_parameters_t kernel_scene_params;
__device__ int kernel_num_spheres, kernel_num_triangles, kernel_num_models, kernel_num_instances;
__device__ sphere_instance_t* kernel_spheres;
__device__ triangle_t* kernel_triangles;
__device__ triangle_extra_t* kernel_extras;
__device__ model_t* kernel_models;
__device__ instance_t* kernel_instances;
__device__ curandState* kernel_curand_state;

__device__ float sphere_t::intersect(vec3 start, vec3 direction) {

	float a = direction.magnitude_2();
	vec3 recentered = start - center;
	float b = 2 * direction.dot(recentered);
	float recentered_radius_2 = recentered.magnitude_2();
	float c = recentered_radius_2 - (radius * radius);

	float discrim = (b * b) - (4.0f * a * c);
	if (discrim < 0.0f) return -1.0f;

	float sqrt_discrim = sqrtf(discrim);
	float t1 = (-b + sqrt_discrim) / (2.0f * a);
	float t2 = (-b - sqrt_discrim) / (2.0f * a);

	float t;
	if (c < 0.0f) {
		t = fmaxf(t1, t2);
	} else {
		t = fminf(t1, t2);
	}
	return t;

}

#define TRI_INT_EPSILON 0.000001f

__device__ float triangle_t::intersect(vec3 start, vec3 direction) {

	vec3 P = direction.cross(ac);
	float det = ab.dot(P);
	if (det > -TRI_INT_EPSILON && det < TRI_INT_EPSILON) return -1.0f;
	float inv_det = 1.0f / det;

	vec3 T = start - a;
	float u = T.dot(P) * inv_det;
	if (u < 0.0f || u > 1.0f) return -1.0f;

	vec3 Q = T.cross(ab);
	float v = Q.dot(direction) * inv_det;
	if (v < 0.0f || u + v > 1.0f) return -1.0f;

	float t = ac.dot(Q) * inv_det;

	return (t > TRI_INT_EPSILON) ? t : -1.0f;

}

__device__ void triangle_t::barycentric(vec3 point, float& u, float& v, float& w) {

	vec3 ah = point - a;

	float ab_ab = ab.dot(ab);
	float ab_ac = ab.dot(ac);
	float ac_ac = ac.dot(ac);
	float ab_ah = ab.dot(ah);
	float ac_ah = ac.dot(ah);

	float inv_denom = 1.0f / (ab_ab * ac_ac - ab_ac * ab_ac);

	v = (ac_ac * ab_ah - ab_ac * ac_ah) * inv_denom;
	w = (ab_ab * ac_ah - ab_ac * ab_ah) * inv_denom;
	u = 1.0f - v - w;

}

__device__ vec3 triangle_extra_t::interpolate_normals(float u, float v, float w) {

	return an * u + bn * v + cn * w;

}

__device__ uv_t triangle_extra_t::interpolate_uvs(float u, float v, float w) {

	return at * u + bt * v + ct * w;

}

__device__ float aabb_t::intersect(vec3 start, vec3 inv_direction) {

	vec3 high_diff = high - start;
	vec3 low_diff = low - start;

	float td1, td2, t_max, t_min;

	// X
	td1 = high_diff.x * inv_direction.x;
	td2 = low_diff.x * inv_direction.x;
	t_max = fmaxf(td1, td2);
	t_min = fminf(td1, td2);

	// Y
	td1 = high_diff.y * inv_direction.y;
	td2 = low_diff.y * inv_direction.y;
	t_max = fminf(t_max, fmaxf(td1, td2));
	t_min = fmaxf(t_min, fminf(td1, td2));

	// Z
	td1 = high_diff.z * inv_direction.z;
	td2 = low_diff.z * inv_direction.z;
	t_max = fminf(t_max, fmaxf(td1, td2));
	t_min = fmaxf(t_min, fminf(td1, td2));

	if (t_max < 0.0f) {
		return -1.0f;
	} else if (t_min < 0.0f) {
		return 0.0f;
	} else if (t_max > t_min) {
		return t_min;
	} else {
		return -1.0f;
	}

}

__device__ vec3 random_isotropic(float cos_theta, int n) {

	float phi = 2.0f * M_PI * curand_uniform(&kernel_curand_state[n]);
	float sin_phi, cos_phi;
	__sincosf(phi, &sin_phi, &cos_phi);

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

	float cos_theta = __powf(curand_uniform(&kernel_curand_state[n]), 1.0f / (spec_power + 1.0f));
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
	float sin_theta, cos_theta;
	__sincosf(theta, &sin_theta, &cos_theta);
	float sqrtr = sqrtf(curand_uniform(&kernel_curand_state[n]));
	return ortho1 * sqrtr * sin_theta + ortho2 * sqrtr * cos_theta;
}

__device__ float intersect_bvh(bvh_node_t* bvh, int tri_start, vec3 ray_direction, vec3 ray_start, int& tri_index) {

	int stack[32];
	int stack_index = 0;
	stack[0] = 0;

	float t = INFINITY;
	int local_tri_index = -1;

	vec3 inv_ray_direction;
	inv_ray_direction.x = 1.0f / ray_direction.x;
	inv_ray_direction.y = 1.0f / ray_direction.y;
	inv_ray_direction.z = 1.0f / ray_direction.z;

	while (stack_index >= 0) {

		bvh_node_t node = bvh[stack[stack_index]];
		stack_index--;

		float bound_t = node.bound.intersect(ray_start, inv_ray_direction);
		if (bound_t < 0.0f || bound_t >= t) continue;

		if (node.i1 & BVH_LEAF_MASK) {

			for (int i = tri_start + node.i0; i < tri_start + (node.i1 & BVH_I1_MASK); i++) {

				float test_t = kernel_triangles[i].intersect(ray_start, ray_direction);
				if (test_t >= 0.0f && test_t < t) {
					t = test_t;
					local_tri_index = i;
				}

			}

		} else {

			stack_index++;
			stack[stack_index] = (node.i1 & BVH_I1_MASK);
			stack_index++;
			stack[stack_index] = node.i0;

		}

	}

	tri_index = local_tri_index;
	return t;

}

__global__ void initRenderKernel(float* output_buffer, color3* render_buffer, curandState* curand_state, sphere_instance_t* spheres, triangle_t* triangles, triangle_extra_t* extras, model_t* models, instance_t* instances, int render_width, int render_height, scene_parameters_t scene_params, int num_spheres, int num_triangles, int num_models, int num_instances) {

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
			kernel_num_triangles = num_triangles;
			kernel_triangles = triangles;
			kernel_extras = extras;
			kernel_num_models = num_models;
			kernel_models = models;
			kernel_num_instances = num_instances;
			kernel_instances = instances;
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

		vec3 dof_confusion = confusion_disk(camera.up, camera.right, n) * camera.aperture_radius;
		vec3 ray_start = camera.position + dof_confusion;
		vec3 ray_direction = (camera.forward + camera.right * camera.aspect_ratio * screen_x + camera.up * screen_y) * camera.focal_distance - dof_confusion;
		ray_direction.normalize();

		color3 final_color = {0.0f, 0.0f, 0.0f};
		color3 running_absorption = {1.0f, 1.0f, 1.0f};

		volume_t cur_volume = kernel_scene_params.air_volume;

		for (int depth = 0; depth < MAX_DEPTH; depth++) {

			float t = INFINITY;
			int obj_index = -1;
			int obj_subindex = -1;
			int obj_type;

			for (int i = 0; i < kernel_num_spheres; i++) {

				float test_t = kernel_spheres[i].shape.intersect(ray_start, ray_direction);

				if (test_t >= 0.0f && test_t < t) {
					t = test_t;
					obj_index = i;
					obj_type = SPHERE_TYPE;
				}

			}

			for (int i = 0; i < kernel_num_instances; i++) {

				instance_t instance = kernel_instances[i];
				model_t model = kernel_models[instance.model_index];

				vec3 trans_ray_start = instance.inv_transform * ray_start;
				vec3 trans_ray_direction = instance.inv_transform.apply_rot(ray_direction);

				int tri_index;
				float test_t = intersect_bvh(model.bvh, model.tri_start, trans_ray_direction, trans_ray_start, tri_index);
				if (tri_index >= 0 && test_t >= 0.0f && test_t < t) {
					t = test_t;
					obj_index = i;
					obj_subindex = tri_index;
					obj_type = INSTANCE_TYPE;
				}

			}

			float scatter_t = (cur_volume.scatter > 0.0f) ? -__logf(curand_uniform(&kernel_curand_state[n])) / cur_volume.scatter : INFINITY;

			if (t > scatter_t) { // scatter

				color3 attenuation = cur_volume.attenuation;
				color3 beer = { // shortcut if any component is zero to get rid of fireflies
					attenuation.r > 0.0f ? __expf(scatter_t * __logf(attenuation.r)) : 0.0f,
					attenuation.g > 0.0f ? __expf(scatter_t * __logf(attenuation.g)) : 0.0f,
					attenuation.b > 0.0f ? __expf(scatter_t * __logf(attenuation.b)) : 0.0f
				};
				running_absorption *= beer;

				ray_start += ray_direction * scatter_t;
				ray_direction = random_henyey_greenstein(cur_volume.scatter_g, n).change_up(ray_direction);

			} else if (obj_index != -1) { // interact with surface

				vec3 surface_position = ray_start + ray_direction * t;

				material_t material;
				vec3 normal, effective_normal;
				color3 effective_diffuse;
				if (obj_type == SPHERE_TYPE) {

					material = kernel_spheres[obj_index].material;

					normal = (surface_position - kernel_spheres[obj_index].shape.center);
					normal.normalize();

					effective_normal = normal;

					effective_diffuse = material.surface.diffuse;

				} else if (obj_type == INSTANCE_TYPE) {

					instance_t& instance = kernel_instances[obj_index];
					material = instance.material;

					vec3 model_surface_position = instance.inv_transform * surface_position; // surface position in model space
					float u, v, w;
					kernel_triangles[obj_subindex].barycentric(model_surface_position, u, v, w);

					normal = kernel_triangles[obj_subindex].ab.cross(kernel_triangles[obj_subindex].ac);
					normal.normalize(); // don't need this when inverse transpose is used (see next comment)
					normal = instance.transform.apply_rot(normal); // really need to apply inverse transpose to scale properly
					normal.normalize();

					if (material.surface.interpolate_normals) {
						effective_normal = kernel_extras[obj_subindex].interpolate_normals(u, v, w);
						effective_normal.normalize(); // don't need this when inverse transpose is used (see next comment)
						effective_normal = instance.transform.apply_rot(effective_normal); // really need to apply inverse transpose to scale properly
						effective_normal.normalize();
					} else {
						effective_normal = normal;
					}

					if (instance.texture != nullptr) {

						uv_t uv = kernel_extras[obj_subindex].interpolate_uvs(u, v, w);

						// temporary texture fetch
						int tex_x = (int) (uv.u * instance.texture_width);
						int tex_y = (int) (uv.v * instance.texture_height);
						color3 tex_color = instance.texture[tex_y * instance.texture_width + tex_x];
						// end temporary texture fetch

						effective_diffuse = tex_color;

					} else {

						effective_diffuse = material.surface.diffuse;

					}

				}
				bool exiting = ray_direction.dot(normal) > 0.0f;
				if (exiting) {
					normal = -normal;
					effective_normal = -effective_normal;
				}

				color3 attenuation = cur_volume.attenuation;
				color3 beer = { // shortcut if any component is zero to get rid of fireflies
					attenuation.r > 0.0f ? __expf(t * __logf(attenuation.r)) : 0.0f,
					attenuation.g > 0.0f ? __expf(t * __logf(attenuation.g)) : 0.0f,
					attenuation.b > 0.0f ? __expf(t * __logf(attenuation.b)) : 0.0f
				};
				running_absorption *= beer;

				ray_start += ray_direction * t;
				vec3 off_surface = normal * 0.0001f; // add small amount to get off the surface (no shading acne)

				float effective_specular_weight = material.surface.specular_weight;

				float n1 = exiting ? material.volume.refractive_index : kernel_scene_params.air_volume.refractive_index;
				float n2 = exiting ? kernel_scene_params.air_volume.refractive_index : material.volume.refractive_index;
				float ni = n1 / n2;

				float cosi = -ray_direction.dot(effective_normal);
				float sint_2 = ni * ni * (1 - cosi * cosi);
				float cost = sqrtf(1 - sint_2);

				if (material.surface.specular_weight > 0.0f) {

					if (sint_2 > 1.0f) {
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
						effective_specular_weight += (1.0f - effective_specular_weight) * r_schlick;
					}
				} else {
					effective_specular_weight = 0.0f;
				}

				if (curand_uniform(&kernel_curand_state[n]) < effective_specular_weight) { // specular

					ray_start += off_surface;

					if (material.surface.spec_power > 0.0f) { // Phong specular

						vec3 ray_reflect = ray_direction.reflect(effective_normal);
						ray_direction = random_phong_hemi(material.surface.spec_power, n).change_up(ray_reflect);
						if (ray_direction.dot(normal) < 0.0f) ray_direction = ray_direction.reflect(normal);

					} else { // perfect reflection

						ray_direction = ray_direction.reflect(effective_normal);
						if (ray_direction.dot(normal) < 0.0f) ray_direction = ray_direction.reflect(normal);

					}

					cur_volume = exiting ? material.volume : kernel_scene_params.air_volume;

					final_color += running_absorption * material.surface.emission;
					running_absorption *= material.surface.specular;

				} else if (curand_uniform(&kernel_curand_state[n]) < material.surface.transmission_weight) { // refract

					ray_start -= off_surface;
					ray_direction = ray_direction * ni + effective_normal * (ni * cosi - cost);
					ray_direction.normalize();
					cur_volume = exiting ? kernel_scene_params.air_volume : material.volume;

				} else { // diffuse

					ray_start += off_surface;
					ray_direction = random_phong_hemi(1.0f, n).change_up(effective_normal);
					if (ray_direction.dot(normal) < 0.0f) ray_direction = ray_direction.reflect(normal);
					cur_volume = exiting ? material.volume : kernel_scene_params.air_volume;

					final_color += running_absorption * material.surface.emission;
					running_absorption *= effective_diffuse;

				}

			} else {

				final_color += running_absorption * kernel_scene_params.background_emission;
				break;
			}

		}

		final_color += kernel_render_buffer[n];
		kernel_render_buffer[n] = final_color;

		color3 output_color = (final_color / render_count).linearToGamma() * 255.0f;
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

	if (cudaDeviceSetCacheConfig(cudaFuncCachePreferL1)) {
		std::cout << "Could not set cache configuration." << std::endl;
		return false;
	}

	if (cudaMalloc(&dev_render_buffer, render_n * sizeof(color3)) != cudaSuccess) {
		std::cout << "Cannot allocate enough GPU memory." << std::endl;
		return false;
	}

	should_clear = true;

	if (cudaMalloc(&dev_spheres, scene.spheres.size() * sizeof(sphere_instance_t)) != cudaSuccess) {
		std::cout << "Cannot allocate enough GPU memory." << std::endl;
		return false;
	}

	if (cudaMemcpy(dev_spheres, scene.spheres.data(), scene.spheres.size() * sizeof(sphere_instance_t), cudaMemcpyHostToDevice) != cudaSuccess) {
		std::cout << "Cannot upload spheres." << std::endl;
		return false;
	}

	if (cudaMalloc(&dev_triangles, scene.triangles.size() * sizeof(triangle_t)) != cudaSuccess) {
		std::cout << "Cannot allocate enough GPU memory." << std::endl;
		return false;
	}

	if (cudaMemcpy(dev_triangles, scene.triangles.data(), scene.triangles.size() * sizeof(triangle_t), cudaMemcpyHostToDevice) != cudaSuccess) {
		std::cout << "Cannot upload triangles." << std::endl;
		return false;
	}

	if (cudaMalloc(&dev_extras, scene.extras.size() * sizeof(triangle_extra_t)) != cudaSuccess) {
		std::cout << "Cannot allocate enough GPU memory." << std::endl;
		return false;
	}

	if (cudaMemcpy(dev_extras, scene.extras.data(), scene.triangles.size() * sizeof(triangle_extra_t), cudaMemcpyHostToDevice) != cudaSuccess) {
		std::cout << "Cannot upload triangle extras." << std::endl;
		return false;
	}

	for (int i = 0; i < scene.bvhs.size(); i++) {

		bvh_node_t* dev_bvh;

		if (cudaMalloc(&dev_bvh, scene.bvhs[i].size() * sizeof(bvh_node_t)) != cudaSuccess) {
			std::cout << "Cannot allocate enough GPU memory." << std::endl;
			return false;
		}

		if (cudaMemcpy(dev_bvh, scene.bvhs[i].data(), scene.bvhs[i].size() * sizeof(bvh_node_t), cudaMemcpyHostToDevice) != cudaSuccess) {
			std::cout << "Cannot upload BVHs." << std::endl;
			return false;
		}

		dev_bvhs.push_back(dev_bvh);
		scene.models[i].bvh = dev_bvh;

	}

	if (cudaMalloc(&dev_models, scene.models.size() * sizeof(model_t)) != cudaSuccess) {
		std::cout << "Cannot allocate enough GPU memory." << std::endl;
		return false;
	}

	if (cudaMemcpy(dev_models, scene.models.data(), scene.models.size() * sizeof(model_t), cudaMemcpyHostToDevice) != cudaSuccess) {
		std::cout << "Cannot upload models." << std::endl;
		return false;
	}

	for (int i = 0; i < scene.textures.size(); i++) {

		color3* dev_texture;

		if (cudaMalloc(&dev_texture, scene.textures[i].data.size() * sizeof(color3)) != cudaSuccess) {
			std::cout << "Cannot allocate enough GPU memory." << std::endl;
			return false;
		}

		if (cudaMemcpy(dev_texture, scene.textures[i].data.data(), scene.textures[i].data.size() * sizeof(color3), cudaMemcpyHostToDevice) != cudaSuccess) {
			std::cout << "Cannot upload textures." << std::endl;
			return false;
		}

		dev_textures.push_back(dev_texture);

	}

	for (int i = 0; i < scene.instances.size(); i++) {

		int texture_index = scene.instances[i].texture_index;
		if (texture_index >= 0) {
			scene.instances[i].texture = dev_textures[texture_index];
			scene.instances[i].texture_width = scene.textures[texture_index].width;
			scene.instances[i].texture_height = scene.textures[texture_index].height;
		} else {
			scene.instances[i].texture = nullptr;
		}

	}

	if (cudaMalloc(&dev_instances, scene.instances.size() * sizeof(instance_t)) != cudaSuccess) {
		std::cout << "Cannot allocate enough GPU memory." << std::endl;
		return false;
	}

	if (cudaMemcpy(dev_instances, scene.instances.data(), scene.instances.size() * sizeof(instance_t), cudaMemcpyHostToDevice) != cudaSuccess) {
		std::cout << "Cannot upload instances." << std::endl;
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
	initRenderKernel<<<blocks, threads_per_block>>>(dev_output_buffer, dev_render_buffer, dev_curand_state, dev_spheres, dev_triangles, dev_extras, dev_models, dev_instances, render_width, render_height, scene.params, (int) scene.spheres.size(), (int) scene.triangles.size(), (int) scene.models.size(), (int) scene.instances.size());

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
		start_time = std::chrono::high_resolution_clock::now();
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

void getRenderStatus(int& _render_count, double& render_time) {

	_render_count = render_count;

	std::chrono::time_point<std::chrono::steady_clock> cur_time = std::chrono::high_resolution_clock::now();
	long long dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cur_time - start_time).count();
	render_time = (double) dt_ms * 0.001;

}
