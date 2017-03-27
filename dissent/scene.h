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
	float refractive_index, scatter;
	color3 attenuation;
};

struct material_t {
	surface_t surface;
	volume_t volume;
};

struct scene_parameters_t {
	volume_t air_volume;
	color3 background_emission;
	float aperture_radius, focal_distance;
};

struct scene_t {
	scene_parameters_t params;
	std::vector<sphere_t> spheres;
	std::vector<material_t> materials;

	void addSphere(vec3 center, float radius) {
		spheres.push_back({center, radius});
		materials.push_back({{
			0.0f,
			0.0f,
			0.0f,
			{0.0f, 0.0f, 0.0f},
			{1.0f, 1.0f, 1.0f},
			{0.0f, 0.0f, 0.0f}
		}, {
			1.0f,
			0.0f,
			{1.0f, 1.0f, 1.0f}
		}});
	}

	void setSpecularWeight(float specular_weight) {
		materials.back().surface.specular_weight = specular_weight;
	}

	void setTransmissionWeight(float transmission_weight) {
		materials.back().surface.transmission_weight = transmission_weight;
	}

	void setSpecularPower(float spec_power) {
		materials.back().surface.spec_power = spec_power;
	}

	void setRefractiveIndex(float refractive_index) {
		materials.back().volume.refractive_index = refractive_index;
	}

	void setDiffuse(color3 diffuse) {
		materials.back().surface.diffuse = diffuse.gammaToLinear();
	}

	void setSpecular(color3 specular) {
		materials.back().surface.specular = specular.gammaToLinear();
	}

	void setEmission(color3 emission, float emission_intensity) {
		materials.back().surface.emission = emission.gammaToLinear() * emission_intensity;
	}

	void setAttenuation(color3 attenuation) {
		materials.back().volume.attenuation = attenuation.gammaToLinear();
	}

	void setScatter(float scatter) {
		materials.back().volume.scatter = scatter;
	}

};
