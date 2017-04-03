#pragma once

#include <vector>

#include "common.h"
#include "geometry.h"

struct builder_triangle_t {
	int av, bv, cv, an, bn, cn, at, bt, ct;
};

class ModelBuilder {

private:
	std::vector<vec3> vertices;
	std::vector<vec3> normals;
	std::vector<builder_triangle_t> triangles;

	int fixIndex(int index, int size);

public:
	void addVertex(vec3 vertex);
	void addNormal(vec3 normal);
	void addTriangle(int av, int bv, int cv, int an, int bn, int cn, int at, int bt, int ct);
	void estimateNormals();
	void extractModel(std::vector<triangle_t>& final_triangles, std::vector<triangle_extra_t>& extras);

};