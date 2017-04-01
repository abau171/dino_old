#pragma once

#include <vector>

#include "common.h"
#include "geometry.h"

struct camera_t {
	vec3 position, forward, up, right;
	float aspect_ratio;
};

struct surface_t {
	float specular_weight, transmission_weight; // first checks for specular reflection, if not then checks for transmission vs diffuse
	float spec_power; // spec_power=0 means perfect reflection
	color3 diffuse, specular, emission;
};

struct volume_t {
	float refractive_index, scatter, scatter_g;
	color3 attenuation;
};

struct material_t {
	surface_t surface;
	volume_t volume;
};

struct sphere_instance_t {
	sphere_t shape;
	material_t material;
};

struct scene_parameters_t {
	volume_t air_volume;
	color3 background_emission;
	float aperture_radius, focal_distance;
};

struct scene_t {
	scene_parameters_t params;
	std::vector<sphere_instance_t> spheres;

	void addSphere(vec3 center, float radius) {
		spheres.push_back({
			{center, radius},
			{{
				0.0f,
				0.0f,
				0.0f,
				{0.0f, 0.0f, 0.0f},
				{1.0f, 1.0f, 1.0f},
				{0.0f, 0.0f, 0.0f}
			}, {
				1.0f,
				0.0f,
				0.0f,
				{1.0f, 1.0f, 1.0f}
			}}});
	}

	void setSpecularWeight(float specular_weight) {
		spheres.back().material.surface.specular_weight = specular_weight;
	}

	void setTransmissionWeight(float transmission_weight) {
		spheres.back().material.surface.transmission_weight = transmission_weight;
	}

	void setSpecularPower(float spec_power) {
		spheres.back().material.surface.spec_power = spec_power;
	}

	void setRefractiveIndex(float refractive_index) {
		spheres.back().material.volume.refractive_index = refractive_index;
	}

	void setDiffuse(color3 diffuse) {
		spheres.back().material.surface.diffuse = diffuse.gammaToLinear();
	}

	void setSpecular(color3 specular) {
		spheres.back().material.surface.specular = specular.gammaToLinear();
	}

	void setEmission(color3 emission, float emission_intensity) {
		spheres.back().material.surface.emission = emission.gammaToLinear() * emission_intensity;
	}

	void setAttenuation(color3 attenuation) {
		spheres.back().material.volume.attenuation = attenuation.gammaToLinear();
	}

	void setScatter(float scatter, float scatter_g = 0.0f) {
		spheres.back().material.volume.scatter = scatter;
		spheres.back().material.volume.scatter_g = scatter_g;
	}

};
