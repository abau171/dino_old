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
	color3 diffuse, specular, emit, attenuation_color;
};

struct scene_parameters_t {
	color3 background_emission;
	float aperture_radius, focal_distance;
};

struct scene_t {
	scene_parameters_t params;
	std::vector<sphere_t> spheres;
	std::vector<material_t> materials;
};
