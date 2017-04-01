#pragma once

#include "common.h"

struct sphere_t {

	vec3 center;
	float radius;

	__device__ float sphere_t::intersect(vec3 start, vec3 direction);

};

struct triangle_t {

	vec3 a, ab, ac;

	__device__ float triangle_t::intersect(vec3 start, vec3 direction);

};