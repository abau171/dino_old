#pragma once

#include <vector>

#include "common.h"
#include "geometry.h"

struct camera_t {
	vec3 position, forward, up, right;
	float aspect_ratio;
};

struct material_t {
	float specular_weight, transmission_weight; // first checks for specular reflection, if not then checks for transmission vs diffuse
	float spec_power, refractive_index; // spec_power=0 means perfect reflection
	color3 diffuse, specular, emission, attenuation;
};

struct scene_parameters_t {
	color3 background_emission;
	float aperture_radius, focal_distance;
};

struct scene_t {
	scene_parameters_t params;
	std::vector<sphere_t> spheres;
	std::vector<material_t> materials;

	void addSphere(vec3 center, float radius) {
		spheres.push_back({center, radius});
		materials.push_back({
			0.0f,
			0.0f,
			0.0f,
			1.0f,
			{0.0f, 0.0f, 0.0f},
			{1.0f, 1.0f, 1.0f},
			{0.0f, 0.0f, 0.0f},
			{0.0f, 0.0f, 0.0f}
		});
	}

	void setSpecularWeight(float specular_weight) {
		materials.back().specular_weight = specular_weight;
	}

	void setTransmissionWeight(float transmission_weight) {
		materials.back().transmission_weight = transmission_weight;
	}

	void setSpecularPower(float spec_power) {
		materials.back().spec_power = spec_power;
	}

	void setRefractiveIndex(float refractive_index) {
		materials.back().refractive_index = refractive_index;
	}

	void setDiffuse(color3 diffuse) {
		materials.back().diffuse = diffuse;
	}

	void setSpecular(color3 specular) {
		materials.back().specular = specular;
	}

	void setEmission(color3 emission) {
		materials.back().emission = emission;
	}

	void setAttenuation(color3 attenuation) {
		materials.back().attenuation = attenuation;
	}

};
