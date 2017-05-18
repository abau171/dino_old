#pragma once

#include <string>
#include <vector>

/*
Load an .obj file into 3D model data.
*/
void loadObj(std::string filename, std::vector<triangle_t>& triangles, std::vector<triangle_extra_t>& extras);
