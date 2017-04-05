#pragma once

#include "GL/glew.h"
#include "GL/glut.h"

struct output_point_t {
	float x, y;
};

struct output_color_t {
	unsigned char r, g, b, a;
};

bool initRender(int width, int height, scene_t& scene, GLuint new_gl_image_buffer);

bool render(camera_t& camera);

void clearRender();

output_color_t* downloadOutputBuffer();

void getRenderStatus(int& _render_count, double& render_time);

float getImageDepth(camera_t& camera, int x, int y);
