#include <iostream>
#include <vector>

#include "common.h"
#include "geometry.h"

#include "builder.h"

/*
Convert an .obj-style index to an absolute index.
*/
int ModelBuilder::fixIndex(int index, int size) {

	if (index >= 0) {
		return index - 1;
	} else {
		return size + index;
	}

}

/*
Add a vertex to the model.
*/
void ModelBuilder::addVertex(vec3 vertex) {

	vertices.push_back(vertex);

}

/*
Add a vertex normal to the model.
*/
void ModelBuilder::addNormal(vec3 normal) {

	normals.push_back(normal);

}

/*
Add a UV coordinate to the model.
*/
void ModelBuilder::addUV(uv_t uv) {

	uvs.push_back(uv);

}

/*
Add a triangle to the model.
This includes 3 each of vertex, vertex normal, and texture coordinate indices.
*/
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

/*
Clears all input vertex normals, and guesses what they should be for each vertex.
*/
void ModelBuilder::estimateNormals() {

	// clear all normals
	normals.resize(vertices.size());
	for (int i = 0; i < normals.size(); i++) {
		normals[i] = {0.0f, 0.0f, 0.0f};
	}

	// set each vertex to the sum of the normals of each adjacent triangle
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

	// normalize the sums to find the final estimate normals
	for (int i = 0; i < normals.size(); i++) {
		normals[i].normalize();
	}

}

/*
Add simple placeholder values for each UV coordinate.
*/
void ModelBuilder::fakeUVs() {

	// add one set of UV coordinates
	uvs.resize(3);
	uvs[0] = {0.0f, 0.0f};
	uvs[1] = {1.0f, 0.0f};
	uvs[2] = {0.0f, 1.0f};

	// give all triangles the same UV coordinates
	for (int i = 0; i < triangles.size(); i++) {

		builder_triangle_t& btri = triangles[i];

		btri.at = 0;
		btri.bt = 1;
		btri.ct = 2;

	}

}

/*
Extract the final 3D model data from the model builder into provided vectors.
*/
void ModelBuilder::extractModel(std::vector<triangle_t>& final_triangles, std::vector<triangle_extra_t>& extras) {

	// if any normals are missing, estimate all normals
	for (int i = 0; i < triangles.size(); i++) {

		builder_triangle_t btri = triangles[i];
		if (btri.an == -1 || btri.bn == -1 || btri.cn == -1) {
			std::cout << "Model does not have vertex normals, so they will be estimated." << std::endl;
			estimateNormals();
			break;
		}

	}

	// if any UV coordinates are missing, fake all UV coordinates
	for (int i = 0; i < triangles.size(); i++) {

		builder_triangle_t btri = triangles[i];
		if (btri.at == -1 || btri.bt == -1 || btri.ct == -1) {
			std::cout << "Model does not have vertex UVs, so they will be faked." << std::endl;
			fakeUVs();
			break;
		}

	}

	// add each triangle and related information to the provided vectors
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
