#pragma once

#include "common.h"

/*
Axis-aligned bounding box structure.
*/
struct aabb_t {

	vec3 low, high;

	/*
	Find the intersection of this bounding box with a ray.
	*/
	__device__ float intersect(vec3 start, vec3 inv_direction);

	/*
	Determine if two bounding boxes overlap.
	*/
	bool overlaps(aabb_t other) {

		return
			(low.x < other.high.x && high.x > other.low.x) &&
			(low.y < other.high.y && high.y > other.low.y) &&
			(low.z < other.high.z && high.z > other.low.z);

	}

	/*
	Determine if this bounding box's centroid is contained in another bounding box.
	*/
	bool centroidWithin(aabb_t other) {

		vec3 centroid = (low + high) / 2.0f;
		return
			(centroid.x >= other.low.x && centroid.x <= other.high.x &&
			centroid.y >= other.low.y && centroid.y <= other.high.y &&
			centroid.z >= other.low.z && centroid.z <= other.high.z);

	}

	/*
	Calculate the surface area of this bounding box.
	*/
	float surface_area() {
		vec3 dim = high - low;
		return 2.0f * (dim.x * dim.y + dim.y * dim.z + dim.z * dim.x);
	}

};

/*
A sphere data structure.
*/
struct sphere_t {

	vec3 center;
	float radius;

	/*
	Determine the intersection of this sphere with a ray.
	*/
	__device__ float intersect(vec3 start, vec3 direction);

};

/*
A triangle structure for triangles in 3D space.
*/
struct triangle_t {

	vec3 a, ab, ac;

	/*
	Determine the intersection of this triangle with a ray.
	*/
	__device__ float intersect(vec3 start, vec3 direction);

	/*
	Find the barycentric coordinates of a coplanar point within a triangle.
	*/
	__device__ void barycentric(vec3 point, float& u, float& v, float& w);

	/*
	Get the boundaries of the triangle.
	*/
	aabb_t getBound() {

		aabb_t bound;
		bound.low.x = fminf(fminf(a.x, a.x + ab.x), a.x + ac.x);
		bound.low.y = fminf(fminf(a.y, a.y + ab.y), a.y + ac.y);
		bound.low.z = fminf(fminf(a.z, a.z + ab.z), a.z + ac.z);
		bound.high.x = fmaxf(fmaxf(a.x, a.x + ab.x), a.x + ac.x);
		bound.high.y = fmaxf(fmaxf(a.y, a.y + ab.y), a.y + ac.y);
		bound.high.z = fmaxf(fmaxf(a.z, a.z + ab.z), a.z + ac.z);

		// small fudge-factor
		vec3 tiny = {0.0001f, 0.0001f, 0.0001f};
		bound.low -= tiny;
		bound.high += tiny;

		return bound;

	}

};

/*
Extra triangle data structure.
*/
struct triangle_extra_t {

	vec3 an, bn, cn;
	uv_t at, bt, ct;

	/*
	Calculate the surface normal through interpolation of vertex normals.
	*/
	__device__ vec3 interpolate_normals(float u, float v, float w);

	/*
	Calculate the true UV value through interpolation.
	*/
	__device__ uv_t interpolate_uvs(float u, float v, float w);

};
