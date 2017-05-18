#pragma once

#include <vector>

#include "common.h"
#include "geometry.h"

/*
Triangle structure used during model construction.
*/
struct builder_triangle_t {
	int av, bv, cv, an, bn, cn, at, bt, ct;
};

/*
ModelBuilder objects can be used to iteratively load data from any source into
a 3D model which can be rendered.
*/
class ModelBuilder {

private:
	std::vector<vec3> vertices;
	std::vector<vec3> normals;
	std::vector<builder_triangle_t> triangles;
	std::vector<uv_t> uvs;

	/*
	Convert an .obj-style index to an absolute index.
	*/
	int fixIndex(int index, int size);

public:

	/*
	Add a vertex to the model.
	*/
	void addVertex(vec3 vertex);

	/*
	Add a vertex normal to the model.
	*/
	void addNormal(vec3 normal);

	/*
	Add a UV coordinate to the model.
	*/
	void addUV(uv_t uv);

	/*
	Add a triangle to the model.
	This includes 3 each of vertex, vertex normal, and texture coordinate indices.
	*/
	void addTriangle(int av, int bv, int cv, int an, int bn, int cn, int at, int bt, int ct);

	/*
	Clears all input vertex normals, and guesses what they should be for each vertex.
	*/
	void estimateNormals();

	/*
	Add simple placeholder values for each UV coordinate.
	*/
	void fakeUVs();

	/*
	Extract the final 3D model data from the model builder into provided vectors.
	*/
	void extractModel(std::vector<triangle_t>& final_triangles, std::vector<triangle_extra_t>& extras);

};