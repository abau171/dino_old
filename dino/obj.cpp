#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"
#include "geometry.h"
#include "builder.h"

#include "obj.h"

static void extractDefinition(std::string definition, int& vertex_index, int& texture_index, int& normal_index) {

	std::istringstream ss(definition);
	char slash;

	ss >> vertex_index;

	if (ss.peek() == '/') {

		ss >> slash;

		if (ss.peek() == '/') {
			texture_index = 0;
		} else {
			ss >> texture_index;
		}

		if (ss.peek() == '/') {

			ss >> slash;

			if ((int) ss.tellg() != -1) {
				ss >> normal_index;
			} else {
				normal_index = 0;
			}

		} else {

			normal_index = 0;

		}

	} else {

		texture_index = 0;
		normal_index = 0;

	}

}

static void processLine(ModelBuilder& builder, std::string line) {

	std::stringstream stream(line);

	std::string type_string;
	stream >> type_string;

	if (type_string.compare("v") == 0) {

		vec3 vertex;
		stream >> vertex.x;
		stream >> vertex.y;
		stream >> vertex.z;

		builder.addVertex(vertex);

	} else if (type_string.compare("vn") == 0) {

		vec3 normal;
		stream >> normal.x;
		stream >> normal.y;
		stream >> normal.z;

		builder.addNormal(normal);

	} else if (type_string.compare("f") == 0) {

		int av, bv, cv, an, bn, cn, at, bt, ct;
		std::string definition;

		stream >> definition;
		extractDefinition(definition, av, at, an);

		stream >> definition;
		extractDefinition(definition, bv, bt, bn);

		stream >> definition;
		extractDefinition(definition, cv, ct, cn);

		builder.addTriangle(av, bv, cv, an, bn, cn, at, bt, ct);

		int remaining = (int) stream.tellg();
		if (remaining != -1) {

			int dv, dn, dt;

			stream >> definition;
			extractDefinition(definition, dv, dt, dn);

			builder.addTriangle(cv, dv, av, cn, dn, an, ct, dt, at);

		}

	}

}

void loadObj(std::string filename, std::vector<triangle_t>& triangles, std::vector<triangle_extra_t>& extras) {

	std::cout << "Loading OBJ file: " << filename << std::endl;

	ModelBuilder builder;

	std::ifstream file(filename);
	if (file.is_open()) {
		std::string line;
		while (getline(file, line)) {
			processLine(builder, line);
		}
		file.close();
	}

	builder.extractModel(triangles, extras);

}