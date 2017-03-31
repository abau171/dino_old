#pragma once

#include "GL/glew.h"
#include "GL/glut.h"

struct output_point_t {
	float x, y;
	unsigned char r, g, b;
};

bool initRender(int width, int height, scene_t& scene, GLuint new_gl_image_buffer);

bool render(camera_t& camera);

void clearRender();

output_point_t* downloadOutputBuffer();
