#pragma once

#include <string>
#include <vector>

#include "common.h"

/*
Texture data structure.
*/
struct texture_t {
	int width, height;
	std::vector<color3> data;
};

/*
Load a texture from a .png file.
*/
bool loadTexture(std::string filename, texture_t& tex);
