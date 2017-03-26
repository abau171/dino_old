#pragma once

#include <vector>

#include "common.h"
#include "geometry.h"

struct camera_t {
	vec3 position, forward, up, right;
	float aspect_ratio;
};

struct surface_t {
	float reflectance, transmission, refractive_index;
	color3 diffuse, specular, emit;
};

struct scene_parameters_t {
	color3 background_emission;
	float aperture_radius, focal_distance;
};

struct scene_t {
	scene_parameters_t params;
	std::vector<sphere_t> spheres;
	std::vector<surface_t> surfaces;
};
