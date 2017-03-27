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

bool paused = false;

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

	scene.params.background_emission = {1.0f, 1.0f, 1.0f};
	scene.params.aperture_radius = 0.0f;
	scene.params.focal_distance = 8.6f;
	scene.params.air_volume = {1.0f, 0.0f, {1.0f, 1.0f, 1.0f}};

	scene.addSphere({0.0f, -1000.0f, 0.0f}, 1000.0f);
	scene.setDiffuse({0.3f, 0.3f, 0.3f});

	scene.addSphere({-0.0f, 2.5f, 1.5f}, 1.5f);
	scene.setSpecularWeight(0.0f);
	scene.setSpecular({1.0f, 1.0f, 1.0f});
	scene.setSpecularPower(1000.0f);
	scene.setTransmissionWeight(1.0f);
	//scene.setDiffuse({0.8f, 0.0f, 0.3f});
	//scene.setRefractiveIndex(1.6f);
	scene.setAttenuation({1.0f, 0.0f, 0.5f});
	scene.setScatter(2.0f);
	
	//scene.addSphere({-2.0f, 60.0f, 2.5f}, 50.0f);
	//scene.setEmission({2.0f, 2.0f, 2.0f});

}

void updateImage() {

	if (!render((unsigned char*) image_data, camera)) return;

}

void display() {

	if (!paused) {
		updateImage();
		glutPostRedisplay();
	}

	glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, image_data);
	glutSwapBuffers();

}

void keyboard(unsigned char key, int x, int y) {

	switch (key) {
	case 'q':
		exit(0);
		break;
	case 'p':
		paused = !paused;
		glutPostRedisplay();
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
