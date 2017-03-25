#include "geometry.h"

__device__ bool sphere_t::intersect(vec3 start, vec3 direction, float& t, vec3& normal) {

	float a = direction.magnitude_2();
	vec3 recentered = start - center;
	float b = 2 * direction.dot(recentered);
	float c = recentered.magnitude_2() - (radius * radius);

	float discrim = (b * b) - (4.0f * a * c);
	if (discrim < 0.0f) return false;

	float sqrt_discrim = std::sqrtf(discrim);
	float t1 = (-b + sqrt_discrim) / (2.0f * a);
	float t2 = (-b - sqrt_discrim) / (2.0f * a);

	if (t1 > 0.0f && t2 > 0.0f) {
		t = std::fminf(t1, t2);
	} else {
		t = std::fmaxf(t1, t2);
	}
	if (t < 0.0f) return false;

	vec3 surface_point = start + direction * t;
	normal = (surface_point - center) / radius;

	return true;

}
