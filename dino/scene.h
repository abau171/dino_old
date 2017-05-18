#pragma once

#include <string>
#include <vector>

#include "common.h"
#include "geometry.h"
#include "obj.h"
#include "bvh.h"
#include "texture.h"

/*
Scene camera data structure.
*/
struct camera_t {

	vec3 position, forward, up, right;
	float phi, theta, aspect_ratio, aperture_radius, focal_distance;

	/*
	Initialize the camera.
	*/
	void init(vec3 new_position, float new_aspect_ratio, float new_aperture_radius = 0.0f, float new_focal_distance = 0.1f) {

		position = new_position;
		aspect_ratio = new_aspect_ratio;

		aperture_radius = 0.0f;
		updateApertureRadius(new_aperture_radius);

		focal_distance = 0.0f;
		updateFocalDistance(new_focal_distance);

		set_rotation(0.0f, 0.0f);

	}

	/*
	Set the direction the camera is pointing.
	*/
	void set_rotation(float new_phi, float new_theta) {

		phi = new_phi;
		theta = new_theta;

		forward = {0.0f, 0.0f, -1.0f};
		forward = mat4_rotate_y(phi) * (mat4_rotate_x(theta) * forward);
		forward.normalize();
		up = {0.0f, 1.0f, 0.0f};
		right = forward.cross(up);
		right.normalize();
		up = right.cross(forward);

	}

	/*
	Rotate the camera.
	*/
	void rotate(float d_phi, float d_theta) {

		set_rotation(phi + d_phi, theta + d_theta);

	}

	/*
	Adjust the camera aperture.
	*/
	void updateApertureRadius(float d_radius) {

		aperture_radius = fmaxf(0.0f, aperture_radius + d_radius);

	}

	/*
	Adjust the camera focal distance.
	*/
	void updateFocalDistance(float d_dist) {

		focal_distance = fmaxf(0.1f, focal_distance + d_dist);

	}

};

/*
Structure describing the surface reflection properties of an object.
*/
struct surface_t {
	float specular_weight, transmission_weight; // first checks for specular reflection, if not then checks for transmission vs diffuse
	float specular_power; // specular_power=0 means perfect reflection
	float transmission_power; // transmission_power=0 means perfect transmission
	color3 diffuse, specular, emission;
	bool interpolate_normals;
};

/*
Structure describing the properties of an object's volume.
*/
struct volume_t {
	float refractive_index, scatter, scatter_g;
	color3 attenuation;
};

/*
Structure describing the physical properties of an object.
*/
struct material_t {
	surface_t surface;
	volume_t volume;
};

/*
Sphere object structure.
*/
struct sphere_instance_t {
	sphere_t shape;
	material_t material;
};

/*
3D model object structure.
*/
struct model_t {
	int tri_start, tri_end;
	bvh_node_t* bvh;
};

/*
Structure describing an instance of a 3D model.
*/
struct instance_t {
	int model_index, texture_index;
	mat4 transform;
	mat4 inv_transform;
	material_t material;
	color3* texture;
	int texture_width, texture_height;
};

/*
Extra scene parameters uploaded to GPU to be used during a render.
*/
struct scene_parameters_t {
	int max_depth;
	volume_t air_volume;
	color3 background_emission;
};

/*
Full scene description structure.
*/
struct scene_t {

	scene_parameters_t params;

	material_t* last_material;
	mat4* last_transform;
	mat4* last_inv_transform;

	std::vector<sphere_instance_t> spheres;
	std::vector<triangle_t> triangles;
	std::vector<triangle_extra_t> extras;
	std::vector<model_t> models;
	std::vector<std::vector<bvh_node_t>> bvhs;
	std::vector<texture_t> textures;
	std::vector<instance_t> instances;

	/*
	Initialize the scene.
	*/
	void init() {

		params.max_depth = 2;
		params.background_emission = {0.0f, 0.0f, 0.0f};
		params.air_volume = {1.0f, 0.0f, 0.0f, {1.0f, 1.0f, 1.0f}};

	}

	/*
	Set the maximum recursion depth of the path tracing algorithm.
	*/
	void setMaxDepth(int max_depth) {

		params.max_depth = max_depth;

	}

	/*
	Set the background emission color.
	*/
	void setBackgroundEmission(color3 emission) {

		params.background_emission = emission.gammaToLinear();

	}

	/*
	Add a sphere to the scene.
	*/
	void addSphere(vec3 center, float radius) {

		material_t material = {{
				0.0f,
				0.0f,
				0.0f,
				0.0f,
				{0.0f, 0.0f, 0.0f},
				{1.0f, 1.0f, 1.0f},
				{0.0f, 0.0f, 0.0f},
				false
			}, {
				1.0f,
				0.0f,
				0.0f,
				{1.0f, 1.0f, 1.0f}
			}};

		spheres.push_back({
			{center, radius},
			material});

		last_material = &spheres.back().material;

	}

	/*
	Load a 3D model from an .obj file into the scene.
	*/
	int addModel(std::string filename) {

		int tri_start = (int) triangles.size();

		std::vector<triangle_t> model_triangles;
		std::vector<triangle_extra_t> model_extras;
		loadObj(filename, model_triangles, model_extras);

		std::vector<indexed_aabb_t> bounds;
		for (int i = 0; i < model_triangles.size(); i++) {
			indexed_aabb_t bound;
			bound.aabb = model_triangles[i].getBound();
			bound.index = i;
			bounds.push_back(bound);
		}

		int bvh_index = (int) bvhs.size();
		bvhs.push_back(std::vector<bvh_node_t>());
		std::vector<int> indices;
		buildBVH(bounds, bvhs[bvh_index], indices);

		for (int i = 0; i < indices.size(); i++) {
			triangles.push_back(model_triangles[indices[i]]);
			extras.push_back(model_extras[indices[i]]);
		}

		models.push_back({tri_start, tri_start + (int) indices.size()});
		return (int) models.size() - 1;

	}

	/*
	Load a texture from a .png file into the scene.
	*/
	int addTexture(std::string filename) {

		texture_t tex;

		if (loadTexture(filename, tex)) {

			textures.push_back(tex);
			return (int) textures.size() - 1;

		} else {

			return -1;

		}

	}

	/*
	Add a new instance of a model to the scene.
	*/
	void addInstance(int model_index, int texture_index=-1) {

		material_t material = {{
				0.0f,
				0.0f,
				0.0f,
				0.0f,
				{0.0f, 0.0f, 0.0f},
				{1.0f, 1.0f, 1.0f},
				{0.0f, 0.0f, 0.0f},
				false
			}, {
				1.0f,
				0.0f,
				0.0f,
				{1.0f, 1.0f, 1.0f}
			}};

		instances.push_back({
			model_index,
			texture_index,
			mat4_identity(),
			mat4_identity(),
			material,
			nullptr});

		last_material = &instances.back().material;
		last_transform = &instances.back().transform;
		last_inv_transform = &instances.back().inv_transform;

	}

	/*
	Set the specular weight of the last added object.
	*/
	void setSpecularWeight(float specular_weight) {
		last_material->surface.specular_weight = specular_weight;
	}

	/*
	Set the transmission weight of the last added object.
	*/
	void setTransmissionWeight(float transmission_weight) {
		last_material->surface.transmission_weight = transmission_weight;
	}

	/*
	Set the specular distribution exponent of the last added object.
	*/
	void setSpecularPower(float specular_power) {
		last_material->surface.specular_power = specular_power;
	}

	/*
	Set the transmission distribution exponent of the last added object.
	*/
	void setTransmissionPower(float transmission_power) {
		last_material->surface.transmission_power = transmission_power;
	}

	/*
	Set the refractive index of the last added object.
	*/
	void setRefractiveIndex(float refractive_index) {
		last_material->volume.refractive_index = refractive_index;
	}

	/*
	Set the diffuse color of the last added object.
	*/
	void setDiffuse(color3 diffuse) {
		last_material->surface.diffuse = diffuse.gammaToLinear();
	}

	/*
	Set the specular color of the last added object.
	*/
	void setSpecular(color3 specular) {
		last_material->surface.specular = specular.gammaToLinear();
	}

	/*
	Set the emission color of the last added object.
	*/
	void setEmission(color3 emission, float emission_intensity) {
		last_material->surface.emission = emission.gammaToLinear() * emission_intensity;
	}

	/*
	Set the attenuation color of the last added object.
	*/
	void setAttenuation(color3 attenuation) {
		last_material->volume.attenuation = attenuation.gammaToLinear();
	}

	/*
	Set the scattering properties of the last added object.
	*/
	void setScatter(float scatter, float scatter_g = 0.0f) {
		last_material->volume.scatter = scatter;
		last_material->volume.scatter_g = scatter_g;
	}

	/*
	Toggle normal interpolation of the last added object.
	*/
	void setInterpolateNormals(bool interpolate) {
		last_material->surface.interpolate_normals = interpolate;
	}

	/*
	Translate the last added object.
	*/
	void translate(vec3 translation) {
		*last_transform = mat4_translation(translation) * (*last_transform);
		*last_inv_transform = last_transform->invert();
	}

	/*
	Scale the last added object.
	*/
	void scale(float scalar) {
		*last_transform = mat4_scale(scalar) * (*last_transform);
		*last_inv_transform = last_transform->invert();
	}

	/*
	Rotate the last added object about the x-axis.
	*/
	void rotate_x(float radians) {
		*last_transform = mat4_rotate_x(radians) * (*last_transform);
		*last_inv_transform = last_transform->invert();
	}

	/*
	Rotate the last added object about the y-axis.
	*/
	void rotate_y(float radians) {
		*last_transform = mat4_rotate_y(radians) * (*last_transform);
		*last_inv_transform = last_transform->invert();
	}

	/*
	Rotate the last added object about the z-axis.
	*/
	void rotate_z(float radians) {
		*last_transform = mat4_rotate_z(radians) * (*last_transform);
		*last_inv_transform = last_transform->invert();
	}

};
