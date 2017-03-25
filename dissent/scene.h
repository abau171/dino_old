#pragma once

#include "common.h"
#include "geometry.h"

struct camera_t {
	vec3 position, forward, up;
	float aspect_ratio;
};
