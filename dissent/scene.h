#pragma once

#include <vector>

#include "common.h"
#include "geometry.h"

struct camera_t {
	vec3 position, forward, up, right;
	float aspect_ratio;
};

struct surface_t {
	float reflectance;
	color3 diffuse, specular, emit;
};

struct scene_t {
	color3 background_emission;
	std::vector<sphere_t> spheres;
	std::vector<surface_t> surfaces;
};
