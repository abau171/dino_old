#include <iostream>
#include <ctime>

#include "GL/glew.h"
#include "GL/glut.h"
#include "lodepng.h"

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

	scene.params.background_emission = {0.1f, 0.15f, 0.3f};
	scene.params.background_emission = scene.params.background_emission.gammaToLinear();
	scene.params.aperture_radius = 0.0f;
	scene.params.focal_distance = 8.6f;
	scene.params.air_volume = {1.0f, 0.0f, 0.0f, {1.0f, 1.0f, 1.0f}};
	scene.params.air_volume.attenuation = scene.params.air_volume.attenuation.gammaToLinear();

	scene.addSphere({0.0f, -1000.0f, 0.0f}, 1000.0f);
	scene.setDiffuse({0.5f, 0.5f, 0.5f});

	scene.addSphere({-1003.0f, 0.0f, 0.0f}, 1000.0f);
	scene.setDiffuse({1.0f, 0.0f, 0.0f});

	scene.addSphere({1003.0f, 0.0f, 0.0f}, 1000.0f);
	scene.setDiffuse({0.0f, 1.0f, 0.0f});

	scene.addSphere({0.0f, 0.0f, 997.0f}, 1000.0f);
	scene.setDiffuse({0.5f, 0.5f, 0.5f});

	scene.addSphere({0.0f, 1006.0f, 0.0f}, 1000.0f);
	scene.setDiffuse({0.5f, 0.5f, 0.5f});
	
	scene.addSphere({-1.5f, 2.0f, -2.0f}, 0.5f);
	scene.setDiffuse({0.5f, 0.5f, 0.5f});

	scene.addSphere({2.0f, 2.0f, 0.0f}, 0.5f);
	scene.setEmission({1.0f, 1.0f, 1.0f}, 20.0f);

}

void updateImage() {

	if (!render((unsigned char*) image_data, camera)) return;

}

void saveImage(bool promptForName) {

	std::vector<unsigned char> image_vector;
	image_vector.resize(WIDTH * HEIGHT * 4);
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
			image_vector[4 * WIDTH * y + 4 * x + 0] = image_data[HEIGHT - y - 1][x][0];
			image_vector[4 * WIDTH * y + 4 * x + 1] = image_data[HEIGHT - y - 1][x][1];
			image_vector[4 * WIDTH * y + 4 * x + 2] = image_data[HEIGHT - y - 1][x][2];
			image_vector[4 * WIDTH * y + 4 * x + 3] = 255;
		}
	}

	std::vector<unsigned char> png;
	unsigned int error = lodepng::encode(png, image_vector, WIDTH, HEIGHT);
	if (error) {
		std::cout << "PNG encode error: " << lodepng_error_text(error) << std::endl;
		return;
	}

	time_t t = time(0);
	struct tm now = *localtime(&t);
	std::string filename =
		std::to_string(now.tm_year + 1900) + "-" +
		std::to_string(now.tm_mon) + "-" +
		std::to_string(now.tm_mday) + "_" +
		std::to_string(now.tm_hour) + "-" +
		std::to_string(now.tm_min) + "-" +
		std::to_string(now.tm_sec);

	if (promptForName) {

		std::cout << "Save file as ?.png: ";

		std::string input_filename;
		std::getline(std::cin, input_filename);
		if (!input_filename.empty()) {
			filename = input_filename;
		}

	}

	filename.append(".png");

	lodepng::save_file(png, filename);
	std::cout << "'" << filename << "' saved." << std::endl;

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
	case 'o':
		saveImage(true);
		break;
	case 'O':
		saveImage(false);
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
