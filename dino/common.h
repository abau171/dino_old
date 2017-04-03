#pragma once

#include <cmath>

#include "cuda_runtime.h"

#define INV_GAMMA_EXP (2.2f)
#define GAMMA_EXP (1.0f / INV_GAMMA_EXP)

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

	__host__ __device__ color3 gammaToLinear() {
		return {
			powf(r, INV_GAMMA_EXP),
			powf(g, INV_GAMMA_EXP),
			powf(b, INV_GAMMA_EXP)
		};
	}

	__host__ __device__ color3 linearToGamma() {
		return {
			powf(r, GAMMA_EXP),
			powf(g, GAMMA_EXP),
			powf(b, GAMMA_EXP)
		};
	}
};

struct mat4 {

	float cells[3][4];

	__host__ __device__ vec3 operator*(vec3 v) {

		vec3 result;
		result.x = cells[0][0] * v.x + cells[0][1] * v.y + cells[0][2] * v.z + cells[0][3];
		result.y = cells[1][0] * v.x + cells[1][1] * v.y + cells[1][2] * v.z + cells[1][3];
		result.z = cells[2][0] * v.x + cells[2][1] * v.y + cells[2][2] * v.z + cells[2][3];

		return result;

	}

	__host__ __device__ mat4 operator*(mat4 other) {

		mat4 result;

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				float sum = 0.0f;
				for (int k = 0; k < 3; k++) {
					sum += cells[i][k] * other.cells[k][j];
				}
				if (j == 3) sum += cells[i][3];
				result.cells[i][j] = sum;
			}
		}

		return result;

	}

	__host__ __device__ mat4 invert() {

		float c00 = cells[0][0];
		float c01 = cells[1][0];
		float c02 = cells[2][0];
		float c10 = cells[0][1];
		float c11 = cells[1][1];
		float c12 = cells[2][1];
		float c20 = cells[0][2];
		float c21 = cells[1][2];
		float c22 = cells[2][2];
		float c30 = cells[0][3];
		float c31 = cells[1][3];
		float c32 = cells[2][3];

		float d00 = c11 * c22 - c12 * c21;
		float d10 = c02 * c21 - c01 * c22;
		float d20 = c01 * c12 - c02 * c11;

		float det = c00 * d00 + c10 * d10 + c20 * d20;

		mat4 inverse;

		inverse.cells[0][0] = d00 / det;
		inverse.cells[0][1] = (c12 * c20 - c10 * c22) / det;
		inverse.cells[0][2] = (c10 * c21 - c11 * c20) / det;
		inverse.cells[1][0] = d10 / det;
		inverse.cells[1][1] = (c00 * c22 - c02 * c20) / det;
		inverse.cells[1][2] = (c01 * c20 - c00 * c21) / det;
		inverse.cells[2][0] = d20 / det;
		inverse.cells[2][1] = (c02 * c10 - c00 * c12) / det;
		inverse.cells[2][2] = (c00 * c11 - c01 * c10) / det;

		inverse.cells[0][3] = -(inverse.cells[0][0] * c30
			+ inverse.cells[0][1] * c31
			+ inverse.cells[0][2] * c32);
		inverse.cells[1][3] = -(inverse.cells[1][0] * c30
			+ inverse.cells[1][1] * c31
			+ inverse.cells[1][2] * c32);
		inverse.cells[2][3] = -(inverse.cells[2][0] * c30
			+ inverse.cells[2][1] * c31
			+ inverse.cells[2][2] * c32);

		return inverse;

	}

	__host__ __device__ mat4 matrix_rot_transpose() {

		mat4 transpose;

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				transpose.cells[i][j] = cells[j][i];
			}
		}

		transpose.cells[0][3] = 0;
		transpose.cells[1][3] = 0;
		transpose.cells[2][3] = 0;

		return transpose;

	}

	__host__ __device__ vec3 apply_rot(vec3 v) {

		vec3 result;
		result.x = cells[0][0] * v.x + cells[0][1] * v.y + cells[0][2] * v.z;
		result.y = cells[1][0] * v.x + cells[1][1] * v.y + cells[1][2] * v.z;
		result.z = cells[2][0] * v.x + cells[2][1] * v.y + cells[2][2] * v.z;

		return result;

	}

};

__host__ __device__ inline mat4 mat4_identity() {

	mat4 matrix = {
		{{1.0f, 0.0f, 0.0f, 0.0f},
		{0.0f, 1.0f, 0.0f, 0.0f},
		{0.0f, 0.0f, 1.0f, 0.0f}}};

	return matrix;

}

__host__ __device__ inline mat4 mat4_translation(vec3 translation) {

	mat4 matrix = {
		{{1.0f, 0.0f, 0.0f, translation.x},
		{0.0f, 1.0f, 0.0f, translation.y},
		{0.0f, 0.0f, 1.0f, translation.z}}};

	return matrix;

}

__host__ __device__ inline mat4 mat4_scale(float scalar) {

	mat4 matrix = {
		{{scalar, 0.0f, 0.0f, 0.0f},
		{0.0f, scalar, 0.0f, 0.0f},
		{0.0f, 0.0f, scalar, 0.0f}}};

	return matrix;

}

__host__ __device__ inline mat4 mat4_rotate_x(float radians) {

	mat4 matrix = {
		{{1.0, 0.0, 0.0, 0.0},
		{0.0, cos(radians), -sin(radians), 0.0},
		{0.0, sin(radians), cos(radians), 0.0}}};

	return matrix;

}

__host__ __device__ inline mat4 mat4_rotate_y(float radians) {

	mat4 matrix = {
		{{cos(radians), 0.0, sin(radians), 0.0},
		{0.0, 1.0, 0.0, 0.0},
		{-sin(radians), 0.0, cos(radians), 0.0}}};

	return matrix;

}

__host__ __device__ inline mat4 mat4_rotate_z(float radians) {

	mat4 matrix = {
		{{cos(radians), -sin(radians), 0.0, 0.0},
		{sin(radians), cos(radians), 0.0, 0.0},
		{0.0, 0.0, 1.0, 0.0}}};

	return matrix;

}

struct uv_t {

	float u, v;

	__host__ __device__ uv_t operator+(uv_t other) {
		return {u + other.u, v + other.v};
	}

	__host__ __device__  uv_t operator+=(uv_t other) {
		u += other.u;
		v += other.v;
		return *this;
	}

	__host__ __device__ uv_t operator*(float scalar) {
		return {scalar * u, scalar * v};
	}

	__host__ __device__ uv_t operator*=(float scalar) {
		u *= scalar;
		v *= scalar;
		return *this;
	}

};
