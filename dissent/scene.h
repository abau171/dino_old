#pragma once

#include <vector>

#include "common.h"
#include "geometry.h"

struct camera_t {
	vec3 position, forward, up, right;
	float aspect_ratio;
};

struct surface_t {
	color3 diffuse, emit;
};

struct scene_t {
	std::vector<sphere_t> spheres;
	std::vector<surface_t> surfaces;
};
