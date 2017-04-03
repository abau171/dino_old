#include <iostream>
#include <vector>

#include "common.h"
#include "geometry.h"

#include "builder.h"

int ModelBuilder::fixIndex(int index, int size) {

	if (index >= 0) {
		return index - 1;
	} else {
		return size + index;
	}

}

void ModelBuilder::addVertex(vec3 vertex) {

	vertices.push_back(vertex);

}

void ModelBuilder::addNormal(vec3 normal) {

	normals.push_back(normal);

}

void ModelBuilder::addUV(uv_t uv) {

	uvs.push_back(uv);

}

void ModelBuilder::addTriangle(int av, int bv, int cv, int an, int bn, int cn, int at, int bt, int ct) {

	triangles.push_back({
		fixIndex(av, (int) vertices.size()),
		fixIndex(bv, (int) vertices.size()),
		fixIndex(cv, (int) vertices.size()),
		fixIndex(an, (int) normals.size()),
		fixIndex(bn, (int) normals.size()),
		fixIndex(cn, (int) normals.size()),
		fixIndex(at, (int) uvs.size()),
		fixIndex(bt, (int) uvs.size()),
		fixIndex(ct, (int) uvs.size())});

}

void ModelBuilder::estimateNormals() {

	normals.resize(vertices.size());

	for (int i = 0; i < normals.size(); i++) {
		normals[i] = {0.0f, 0.0f, 0.0f};
	}

	for (int i = 0; i < triangles.size(); i++) {

		builder_triangle_t& btri = triangles[i];

		vec3 ab = vertices[btri.bv] - vertices[btri.av];
		vec3 ac = vertices[btri.cv] - vertices[btri.av];

		vec3 true_normal = ab.cross(ac);
		if (true_normal.magnitude_2() == 0.0f) continue;
		true_normal.normalize();

		normals[btri.av] += true_normal;
		normals[btri.bv] += true_normal;
		normals[btri.cv] += true_normal;

		btri.an = btri.av;
		btri.bn = btri.bv;
		btri.cn = btri.cv;

	}

	for (int i = 0; i < normals.size(); i++) {
		normals[i].normalize();
	}

}

void ModelBuilder::fakeUVs() {

	uvs.resize(3);
	uvs[0] = {0.0f, 0.0f};
	uvs[1] = {1.0f, 0.0f};
	uvs[2] = {0.0f, 1.0f};

	for (int i = 0; i < triangles.size(); i++) {

		builder_triangle_t& btri = triangles[i];

		btri.at = 0;
		btri.bt = 1;
		btri.ct = 2;

	}

}

void ModelBuilder::extractModel(std::vector<triangle_t>& final_triangles, std::vector<triangle_extra_t>& extras) {

	for (int i = 0; i < triangles.size(); i++) {

		builder_triangle_t btri = triangles[i];
		if (btri.an == -1 || btri.bn == -1 || btri.cn == -1) {
			std::cout << "Model does not have vertex normals, so they will be estimated." << std::endl;
			estimateNormals();
			break;
		}

	}

	for (int i = 0; i < triangles.size(); i++) {

		builder_triangle_t btri = triangles[i];
		if (btri.at == -1 || btri.bt == -1 || btri.ct == -1) {
			std::cout << "Model does not have vertex UVs, so they will be faked." << std::endl;
			fakeUVs();
			break;
		}

	}

	for (int i = 0; i < triangles.size(); i++) {

		builder_triangle_t btri = triangles[i];

#ifdef SPACERS
		vec3 _a = vertices[btri.av];
		vec3 _b = vertices[btri.bv];
		vec3 _c = vertices[btri.cv];

		vec3 a = _a * 0.98f + _b * 0.01f + _c * 0.01f;
		vec3 b = _b * 0.98f + _a * 0.01f + _c * 0.01f;
		vec3 c = _c * 0.98f + _b * 0.01f + _a * 0.01f;
#else
		vec3 a = vertices[btri.av];
		vec3 b = vertices[btri.bv];
		vec3 c = vertices[btri.cv];
#endif

		triangle_t tri;
		tri.a = a;
		tri.ab = b - a;
		tri.ac = c - a;

		triangle_extra_t extra;
		extra.an = normals[btri.an];
		extra.bn = normals[btri.bn];
		extra.cn = normals[btri.cn];
		extra.at = uvs[btri.at];
		extra.bt = uvs[btri.bt];
		extra.ct = uvs[btri.ct];

		final_triangles.push_back(tri);
		extras.push_back(extra);

	}

}
