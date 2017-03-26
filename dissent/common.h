#pragma once

#include <cmath>

#include "cuda_runtime.h"

struct vec3 {

	float x, y, z;

	__host__ __device__ vec3 operator+(vec3 other) {
		return {x + other.x, y + other.y, z + other.z};
	}

	__host__ __device__  vec3 operator+=(vec3 other) {
		x += other.x;
		y += other.y;
		z += other.z;
		return *this;
	}

	__host__ __device__ vec3 operator-() {
		return {-x, -y, -z};
	}

	__host__ __device__ vec3 operator-(vec3 other) {
		return {x - other.x, y - other.y, z - other.z};
	}

	__host__ __device__ vec3 operator-=(vec3 other) {
		x -= other.x;
		y -= other.y;
		z -= other.z;
		return *this;
	}

	__host__ __device__ vec3 operator*(float scalar) {
		return {scalar * x, scalar * y, scalar * z};
	}

	__host__ __device__ vec3 operator*=(float scalar) {
		x *= scalar;
		y *= scalar;
		z *= scalar;
		return *this;
	}

	// these two can be dangerous: they are easy to mix with dot product

	//__host__ __device__ vec3 operator*(vec3 other) {
	//	return {other.x * x, other.y * y, other.z * z};
	//}

	//__host__ __device__ vec3 operator*=(vec3 other) {
	//	x *= other.x;
	//	y *= other.y;
	//	z *= other.z;
	//	return *this;
	//}

	__host__ __device__ vec3 operator/(float scalar) {
		return {x / scalar, y / scalar, z / scalar};
	}

	__host__ __device__ vec3 operator/=(float scalar) {
		x /= scalar;
		y /= scalar;
		z /= scalar;
		return *this;
	}

	__host__ __device__ float magnitude_2() {
		return x * x + y * y + z * z;
	}

	__host__ __device__ float magnitude() {
		return std::sqrtf(x * x + y * y + z * z);
	}

	__host__ __device__ void normalize() {
		float mag = magnitude();
		x /= mag;
		y /= mag;
		z /= mag;
	}

	__host__ __device__ float dot(vec3 other) {
		return x * other.x + y * other.y + z * other.z;
	}

	__host__ __device__ vec3 cross(vec3 other) {
		return {
			(y * other.z) - (z * other.y),
			(z * other.x) - (x * other.z),
			(x * other.y) - (y * other.x)
		};
	};

	__host__ __device__ vec3 reflect(vec3 normal) {
		return *this - normal * (2.0f * this->dot(normal));
	}

	__host__ __device__ vec3 change_coord_system(vec3 cx, vec3 cy, vec3 cz) {
		return cx * x + cy * y + cz * z;
	}

	__host__ __device__ vec3 change_up(vec3 cy) {
		// cy assumed normalized
		vec3 cx = cy.cross({0.0f, 1.0f, 0.0f});
		cx.normalize();
		vec3 cz = cx.cross(cy);
		return this->change_coord_system(cx, cy, cz);
	}

};

struct color3 {

	float r, g, b;

	__host__ __device__ color3 operator+(color3 other) {
		return {r + other.r, g + other.g, b + other.b};
	}

	__host__ __device__ color3 operator+=(color3 other) {
		r += other.r;
		g += other.g;
		b += other.b;
		return *this;
	}

	__host__ __device__ color3 operator-() {
		return {-r, -g, -b};
	}

	__host__ __device__ color3 operator-(color3 other) {
		return {r - other.r, g - other.g, b - other.b};
	}

	__host__ __device__ color3 operator-=(color3 other) {
		r -= other.r;
		g -= other.g;
		b -= other.b;
		return *this;
	}

	__host__ __device__ color3 operator*(float scalar) {
		return {scalar * r, scalar * g, scalar * b};
	}

	__host__ __device__ color3 operator*=(float scalar) {
		r *= scalar;
		g *= scalar;
		b *= scalar;
		return *this;
	}

	__host__ __device__ color3 operator*(color3 other) {
		return {other.r * r, other.g * g, other.b * b};
	}

	__host__ __device__ color3 operator*=(color3 other) {
		r *= other.r;
		g *= other.g;
		b *= other.b;
		return *this;
	}

	__host__ __device__ color3 operator/(float scalar) {
		return {r / scalar, g / scalar, b / scalar};
	}

	__host__ __device__ color3 operator/=(float scalar) {
		r /= scalar;
		g /= scalar;
		b /= scalar;
		return *this;
	}

};
