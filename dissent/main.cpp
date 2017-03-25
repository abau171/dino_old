#include <iostream>

#include "GL/glew.h"
#include "GL/glut.h"

#include "common.h"
#include "geometry.h"
#include "scene.h"
#include "render.h"

static const int WIDTH = 640;
static const int HEIGHT = 480;

static unsigned char image_data[HEIGHT][WIDTH][3];

static camera_t camera;
static scene_t scene;

void initScene() {

	vec3 position = {0.0f, 4.0f, 7.0f};

	vec3 lookat = {0.0f, 2.11f, 0.0f};
	vec3 forward = (lookat - position);
	forward.normalize();
	vec3 up = {0.0f, 1.0f, 0.0f};
	vec3 right = forward.cross(up);
	right.normalize();
	up = right.cross(forward);

	camera = {
		position,
		forward,
		up,
		right,
		(float) WIDTH / HEIGHT
	};

	scene.background_emission = {0.7f, 0.9f, 1.4f};

	scene.spheres.push_back({{0.0f, -1000.0f, 0.0f}, 1000.0f});
	scene.surfaces.push_back({0.0f, 0.0f, 1.0f, {0.3f, 0.3f, 0.3f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f}});

	scene.spheres.push_back({{0.5f, 1.5f, -1.5f}, 1.5f});
	scene.surfaces.push_back({0.1f, 1.0f, 1.75f, {0.2f, 0.4f, 0.8f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f}});

	/*scene.spheres.push_back({{-3.0f, 1.0f, 0.5f}, 1.0f});
	scene.surfaces.push_back({0.0f, 0.0f, 1.0f, {0.6f, 0.6f, 0.2f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f}});

	scene.spheres.push_back({{-1.0f, 3.0f, -5.5f}, 3.0f});
	scene.surfaces.push_back({0.4f, 0.0f, 1.0f, {0.2f, 0.4f, 0.8f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f}});

	scene.spheres.push_back({{4.5f, 2.0f, -3.5f}, 2.0f});
	scene.surfaces.push_back({0.7f, 0.0f, 1.0f, {0.9f, 0.5f, 0.1f}, {0.9f, 0.5f, 0.1f}, {0.0f, 0.0f, 0.0f}});*/

	scene.spheres.push_back({{-2.0f, 3.0f, -1.0f}, 1.0f});
	scene.surfaces.push_back({0.0f, 0.0f, 1.0f, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {8.0f, 8.0f, 8.0f}});

	/*scene.spheres.push_back({{2.0f, 3.0f, -4.0f}, 0.5f});
	scene.surfaces.push_back({0.0f, 0.0f, 1.0f, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {20.0f, 5.0f, 5.0f}});

	scene.spheres.push_back({{2.0f, 0.5f, 2.0f}, 0.5f});
	scene.surfaces.push_back({0.0f, 0.0f, 1.0f, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {5.0f, 20.0f, 5.0f}});*/

}

void updateImage() {

	if (!render((unsigned char*) image_data, camera)) return;

}

void display() {

	updateImage();

	glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, image_data);
	glutSwapBuffers();
	glutPostRedisplay();

}

void keyboard(unsigned char key, int x, int y) {

	switch (key) {
	case 'q':
		exit(0);
		break;
	case 'r':
		clearRender();
		break;
	case 'w':
		camera.position += camera.forward;
		clearRender();
		break;
	case 'a':
		camera.position -= camera.right;
		clearRender();
		break;
	case 's':
		camera.position -= camera.forward;
		clearRender();
		break;
	case 'd':
		camera.position += camera.right;
		clearRender();
		break;
	}

}

int main(int argc, char** argv) {

	initScene();
	if (!resetRender(WIDTH, HEIGHT, scene)) return 1;

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(WIDTH, HEIGHT);
	glutCreateWindow("Dissent Path Tracer");

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);

	glutMainLoop();

	return 0;

}
