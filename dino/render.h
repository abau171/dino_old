#pragma once

#include "GL/glew.h"
#include "GL/glut.h"

/*
Screen-space point used for drawing to the OpenGL window.
*/
struct output_point_t {
	float x, y;
};

/*
Output color struct for drawing to the OpenGL window.
*/
struct output_color_t {
	unsigned char r, g, b, a;
};

/*
Initialize the renderer. This should only be called once.
*/
bool initRender(int width, int height, scene_t& scene, GLuint new_gl_image_buffer);

/*
Runs a single pass of the renderer, updating the OpenGL image buffer.
Multiple calls will refine the image by reducing noise.
*/
bool render(camera_t& camera);

/*
Clear the current render state on the next render() call.
*/
void clearRender();

/*
Download the image from the GPU into a buffer in main memory.
*/
output_color_t* downloadOutputBuffer();

/*
Get the number of render cycles and total elapsed render time.
*/
void getRenderStatus(int& _render_count, double& render_time);

/*
Get the apparent distance of an object in the scene at a given pixel location.
*/
float getImageDepth(camera_t& camera, int x, int y);
