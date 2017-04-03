#pragma once

#include <string>
#include <vector>

#include "common.h"

struct texture_t {
	int width, height;
	std::vector<color3> data;
};

bool loadTexture(std::string filename, texture_t& tex);
