#pragma once

#include "common.h"

struct aabb_t {

	vec3 low, high;

	__device__ float aabb_t::intersect(vec3 start, vec3 direction);

	bool overlaps(aabb_t other) {

		return
			(low.x < other.high.x && high.x > other.low.x) &&
			(low.y < other.high.y && high.y > other.low.y) &&
			(low.z < other.high.z && high.z > other.low.z);

	}

	float surface_area() {
		vec3 dim = high - low;
		return 2.0f * (dim.x * dim.y + dim.y * dim.z + dim.z * dim.x);
	}

};

struct sphere_t {

	vec3 center;
	float radius;

	__device__ float sphere_t::intersect(vec3 start, vec3 direction);

};

struct triangle_t {

	vec3 a, ab, ac;

	__device__ float triangle_t::intersect(vec3 start, vec3 direction);

	aabb_t getBound() {
		aabb_t bound;
		bound.low.x = fminf(fminf(a.x, a.x + ab.x), a.x + ac.x);
		bound.low.y = fminf(fminf(a.y, a.y + ab.y), a.y + ac.y);
		bound.low.z = fminf(fminf(a.z, a.z + ab.z), a.z + ac.z);
		bound.high.x = fmaxf(fmaxf(a.x, a.x + ab.x), a.x + ac.x);
		bound.high.y = fmaxf(fmaxf(a.y, a.y + ab.y), a.y + ac.y);
		bound.high.z = fmaxf(fmaxf(a.z, a.z + ab.z), a.z + ac.z);
		return bound;
	}

};
