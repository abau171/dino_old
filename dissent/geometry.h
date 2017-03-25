#pragma once

#include "common.h"

struct sphere_t {

	vec3 center;
	float radius;

	__device__ bool intersect(vec3 start, vec3 direction, float& t, vec3& normal);

};
