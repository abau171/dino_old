#include <string>
#include <vector>
#include <iostream>
#include "lodepng.h"

#include "common.h"

#include "texture.h"

bool loadTexture(std::string filename, texture_t& tex) {

	std::vector<unsigned char> raw_data;
	unsigned int width, height;

	unsigned int error = lodepng::decode(raw_data, width, height, filename);
	if (error) {
		std::cout << "PNG encode error: " << lodepng_error_text(error) << std::endl;
		return false;
	}

	tex.width = (int) width;
	tex.height = (int) height;

	tex.data.resize(tex.width * tex.height);

	for (int j = 0; j < tex.height; j++) {
		for (int i = 0; i < tex.width; i++) {

			int n = (tex.height - j - 1) * tex.width + i;

			color3 color;
			color.r = raw_data[4 * n + 0] / 255.0;
			color.g = raw_data[4 * n + 1] / 255.0;
			color.b = raw_data[4 * n + 2] / 255.0;

			tex.data[j * tex.width + i] = color.gammaToLinear();

		}
	}

	return true;

}
