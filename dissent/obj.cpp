#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"
#include "geometry.h"

struct obj_load_state_t {
	std::vector<vec3> vertices;
	std::vector<triangle_t> triangles;
};

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

	} else if (type_string.compare("f") == 0) {

		int vertex_index;
		stream >> vertex_index;
		vec3 a = load_state.vertices[vertex_index - 1];
		stream >> vertex_index;
		vec3 b = load_state.vertices[vertex_index - 1];
		stream >> vertex_index;
		vec3 c = load_state.vertices[vertex_index - 1];

		triangle_t triangle;
		triangle.a = a;
		triangle.ab = b - a;
		triangle.ac = c - a;

		load_state.triangles.push_back(triangle);

	}

}

std::vector<triangle_t> loadObj(std::string filename) {

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

	return load_state.triangles;

}