#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"
#include "geometry.h"

struct obj_load_state_t {
	std::vector<vec3> vertices;
	std::vector<vec3> normals;
	std::vector<triangle_t> triangles;
	std::vector<triangle_extra_t> extras;
};

static void extractDefinition(std::string definition, int& vertex_index, int& texture_index, int& normal_index) {

	std::istringstream ss(definition);
	char slash;

	ss >> vertex_index;
	ss >> slash;
	ss >> texture_index;
	ss >> slash;
	ss >> normal_index;

}

static void processLine(obj_load_state_t& load_state, std::string line) {

	std::stringstream stream(line);

	std::string type_string;
	stream >> type_string;

	if (type_string.compare("v") == 0) {

		vec3 vertex;
		stream >> vertex.x;
		stream >> vertex.y;
		stream >> vertex.z;

		load_state.vertices.push_back(vertex);

	} else if (type_string.compare("vn") == 0) {

		vec3 normal;
		stream >> normal.x;
		stream >> normal.y;
		stream >> normal.z;

		load_state.normals.push_back(normal);

	} else if (type_string.compare("f") == 0) {

		int vertex_index, texture_index, normal_index;
		std::string definition;

		stream >> definition;
		extractDefinition(definition, vertex_index, texture_index, normal_index);
		vec3 a = load_state.vertices[vertex_index - 1];
		vec3 an = load_state.normals[normal_index - 1];

		stream >> definition;
		extractDefinition(definition, vertex_index, texture_index, normal_index);
		vec3 b = load_state.vertices[vertex_index - 1];
		vec3 bn = load_state.normals[normal_index - 1];

		stream >> definition;
		extractDefinition(definition, vertex_index, texture_index, normal_index);
		vec3 c = load_state.vertices[vertex_index - 1];
		vec3 cn = load_state.normals[normal_index - 1];

		triangle_t triangle;
		triangle.a = a;
		triangle.ab = b - a;
		triangle.ac = c - a;

		triangle_extra_t extra;
		extra.an = an;
		extra.bn = bn;
		extra.cn = cn;

		load_state.triangles.push_back(triangle);
		load_state.extras.push_back(extra);

		int remaining = (int) stream.tellg();
		if (remaining != -1) {

			stream >> definition;
			extractDefinition(definition, vertex_index, texture_index, normal_index);
			vec3 d = load_state.vertices[vertex_index - 1];
			vec3 dn = load_state.normals[normal_index - 1];

			triangle_t triangle;
			triangle.a = c;
			triangle.ab = d - c;
			triangle.ac = a - c;

			triangle_extra_t extra;
			extra.an = cn;
			extra.bn = dn;
			extra.cn = an;

			load_state.triangles.push_back(triangle);
			load_state.extras.push_back(extra);

		}

	}

}

void loadObj(std::string filename, std::vector<triangle_t>& triangles, std::vector<triangle_extra_t>& extras) {

	obj_load_state_t load_state;

	std::ifstream file(filename);
	if (file.is_open()) {
		std::string line;
		while (getline(file, line)) {
			processLine(load_state, line);
		}
		file.close();
	}

	for (int i = 0; i < load_state.vertices.size(); i++) {
		vec3 vertex = load_state.vertices[i];
	}

	triangles = load_state.triangles;
	extras = load_state.extras;

}