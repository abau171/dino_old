#include <iostream>
#include <ctime>
#include "GL/glew.h"
#include "GL/glut.h"
#include "lodepng.h"

#include "common.h"
#include "geometry.h"
#include "scene.h"
#include "render.h"

#define TEAPOT
#define CORNELL_BOX

static const int WIDTH = 640;
static const int HEIGHT = 480;

GLuint gl_image_buffer;

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

	scene.params.background_emission = {0.4f, 0.6f, 0.9f};
	scene.params.background_emission = scene.params.background_emission.gammaToLinear();
	scene.params.aperture_radius = 0.0f;
	scene.params.focal_distance = 8.6f;
	scene.params.air_volume = {1.0f, 0.0f, 0.0f, {1.0f, 1.0f, 1.0f}};
	scene.params.air_volume.attenuation = scene.params.air_volume.attenuation.gammaToLinear();

#ifdef VI
	int car_model_index = scene.addModel("vi.obj");
	scene.addInstance(car_model_index);
	scene.setDiffuse({1.0f, 0.2f, 0.4f});
	scene.scale(0.03f);
	scene.translate({0.0f, -1.0f, 0.0f});
	scene.rotate_y(-0.3f);

	scene.addSphere({0.0f, -1001.0f, 0.0f}, 1000.0f);
	scene.setDiffuse({0.2f, 0.2f, 0.2f});

	scene.addSphere({0.0f, 90.0f, 0.0f}, 80.0f);
	scene.setEmission({1.0f, 0.9f, 0.8f}, 2.0f);
#endif

#ifdef CAR
	int car_model_index = scene.addModel("car.obj");
	scene.addInstance(car_model_index);
	scene.setDiffuse({1.0f, 0.0f, 0.0f});
	scene.setSpecularWeight(0.05f);
	scene.scale(0.01f);
	scene.translate({0.0f, 1.0f, 0.0f});
	scene.rotate_y(-0.8f);

	scene.addSphere({0.0f, -999.0f, 0.0f}, 1000.0f);
	scene.setDiffuse({0.2f, 0.6f, 0.1f});

	scene.addSphere({-50.0f, 80.0f, 0.0f}, 80.0f);
	scene.setEmission({1.0f, 0.9f, 0.8f}, 2.0f);
#endif

#ifdef TEAPOT
	int teapot_model_index = scene.addModel("teapot.obj");

	scene.addInstance(teapot_model_index);
	scene.setDiffuse({0.8f, 0.5f, 0.0f});
	scene.scale(0.02f);
	scene.rotate_y(-0.8f);
	scene.translate({0.7f, 0.8f, 0.0f});

	scene.addSphere({0.0f, 4.0f, 0.0f}, 1.0f);
	scene.setEmission({1.0f, 1.0f, 1.0f}, 10.0f);

#ifdef OTHER_TEAPOT
	scene.addInstance(teapot_model_index);
	scene.setDiffuse({0.8f, 0.0f, 0.5f});
	scene.scale(0.01f);
	scene.rotate_y(3.9416f);
	scene.translate({-1.2f, 0.8f, 0.0f});
#endif

#endif

#ifdef CORNELL_BOX
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
#endif

}

void saveImage(bool promptForName) {

	output_color_t* output_buffer = downloadOutputBuffer();

	std::vector<unsigned char> image_vector;
	image_vector.resize(WIDTH * HEIGHT * 4);
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
			int n = HEIGHT * x + (HEIGHT - y - 1);
			image_vector[4 * WIDTH * y + 4 * x + 0] = output_buffer[n].r;
			image_vector[4 * WIDTH * y + 4 * x + 1] = output_buffer[n].g;
			image_vector[4 * WIDTH * y + 4 * x + 2] = output_buffer[n].b;
			image_vector[4 * WIDTH * y + 4 * x + 3] = output_buffer[n].a;
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

void tick(int) {

	if (!paused) {
		render(camera);
		glutPostRedisplay();
	}

	glutTimerFunc(1, tick, 0);

}

void display() {

	glClear(GL_COLOR_BUFFER_BIT);

	glBindBuffer(GL_ARRAY_BUFFER, gl_image_buffer);

	glVertexPointer(2, GL_FLOAT, sizeof(output_point_t), (GLvoid*) (WIDTH * HEIGHT * sizeof(float)));
	glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(output_color_t), (GLvoid*) 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, WIDTH * HEIGHT);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

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

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(WIDTH, HEIGHT);
	glutCreateWindow("Dissent Path Tracer");

	glewInit();

	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0f, WIDTH, 0.0f, HEIGHT);

	glGenBuffers(1, &gl_image_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, gl_image_buffer);
	glBufferData(GL_ARRAY_BUFFER, WIDTH * HEIGHT * sizeof(output_color_t) + WIDTH * HEIGHT * sizeof(output_point_t), nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	if (!initRender(WIDTH, HEIGHT, scene, gl_image_buffer)) return 1;

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);

	tick(0);

	glutMainLoop();

	return 0;

}
