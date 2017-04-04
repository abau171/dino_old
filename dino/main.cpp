#include <iostream>
#include <ctime>
#include <chrono>
#include "GL/glew.h"
#include "GL/glut.h"
#include "lodepng.h"

#include "common.h"
#include "geometry.h"
#include "scene.h"
#include "render.h"

#define SKULL

static const int WIDTH = 1280;
static const int HEIGHT = 720;

static GLuint gl_image_buffer;

static camera_t camera;
static scene_t scene;

static std::chrono::time_point<std::chrono::steady_clock> last_time;

static const float MOVEMENT_SPEED = 5.0f;
static const float TURN_SPEED = 0.002f;
static bool left_mouse = false;
static bool right_mouse = false;
static int mouse_x, mouse_y;
static bool w_key = false;
static bool a_key = false;
static bool s_key = false;
static bool d_key = false;
static bool r_key = false;
static bool f_key = false;
static bool t_key = false;
static bool g_key = false;
static bool y_key = false;
static bool h_key = false;

static bool do_clear = false;
static bool paused = false;

void initScene() {

	camera.init({0.0f, 4.0f, 7.0f}, (float) WIDTH / HEIGHT);
	camera.set_rotation(0.0f, -0.25f);

	scene.params.background_emission = {0.4f, 0.6f, 0.9f};
	scene.params.background_emission = scene.params.background_emission.gammaToLinear();
	scene.params.air_volume = {1.0f, 0.0f, 0.0f, {1.0f, 1.0f, 1.0f}};
	scene.params.air_volume.attenuation = scene.params.air_volume.attenuation.gammaToLinear();

#ifdef VI
	int vi_model_index = scene.addModel("vi.obj");
	int vi_texture_index = scene.addTexture("vi.png");
	scene.addInstance(vi_model_index, vi_texture_index);
	scene.setInterpolateNormals(true);
	scene.setDiffuse({1.0f, 0.2f, 0.4f});
	scene.scale(0.03f);
	scene.translate({0.0f, -0.99f, 0.0f});

	scene.addSphere({0.0f, -1001.0f, 0.0f}, 1000.0f);
	scene.setDiffuse({1.0f, 1.0f, 1.0f});
	scene.setSpecularWeight(0.95f);
	scene.setSpecularPower(10000.0f);

	scene.addSphere({0.0f, 30.0f, 50.0f}, 30.0f); // 5.0f
	scene.setEmission({1.0f, 0.8f, 0.6f}, 2.0f); // 50.0f
#endif

#ifdef SKULL
	int skull_model_index = scene.addModel("skull.obj");
	scene.addInstance(skull_model_index);
	scene.setInterpolateNormals(true);
	scene.setDiffuse({1.0f, 1.0f, 1.0f});
	//scene.setTransmissionWeight(1.0f); // uncomment for SSS
	scene.setScatter(20.0f);
	scene.scale(3.0f);
	scene.translate({0.0f, 2.1f, 0.0f});

	scene.addSphere({0.0f, -1001.0f, 0.0f}, 1000.0f);
	scene.setDiffuse({0.3f, 0.3f, 0.3f});

	scene.addSphere({0.0f, 30.0f, 50.0f}, 30.0f); // 5.0f
	scene.setEmission({1.0f, 0.8f, 0.6f}, 2.0f); // 50.0f
#endif

#ifdef CAR
	int car_model_index = scene.addModel("car.obj");
	scene.addInstance(car_model_index);
	scene.setDiffuse({1.0f, 0.0f, 0.0f});
	scene.setSpecularWeight(0.05f);
	scene.setSpecularPower(10000.0f);
	scene.setInterpolateNormals(true);
	scene.scale(0.01f);
	scene.translate({0.0f, 1.0f, 0.0f});
	scene.rotate_y(-0.8f);

	scene.addSphere({0.0f, -999.0f, 0.0f}, 1000.0f);
	scene.setDiffuse({0.2f, 0.4f, 0.1f});

	scene.addSphere({-50.0f, 80.0f, 0.0f}, 20.0f);
	scene.setEmission({1.0f, 0.9f, 0.8f}, 32.0f);
#endif

#ifdef TEAPOT
	int teapot_model_index = scene.addModel("teapot.obj");

	scene.addInstance(teapot_model_index);
	scene.setDiffuse({0.8f, 0.5f, 0.0f});
	scene.setInterpolateNormals(true);
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
	struct tm now;
	localtime_s(&now, &t);
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

static void cameraUpdate() {

	auto cur_time = std::chrono::high_resolution_clock::now();
	long long dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cur_time - last_time).count();
	last_time = cur_time;

	float dt = dt_ms * 0.001f;
	float speed = dt * MOVEMENT_SPEED;

	if (w_key) {
		camera.position += camera.forward * speed;
		do_clear = true;
	}
	if (a_key) {
		camera.position -= camera.right * speed;
		do_clear = true;
	}
	if (s_key) {
		camera.position -= camera.forward * speed;
		do_clear = true;
	}
	if (d_key) {
		camera.position += camera.right * speed;
		do_clear = true;
	}
	if (r_key) {
		camera.position.y += speed;
		do_clear = true;
	}
	if (f_key) {
		camera.position.y -= speed;
		do_clear = true;
	}
	if (t_key) {
		camera.updateFocalDistance(speed);
		do_clear = true;
	}
	if (g_key) {
		camera.updateFocalDistance(-speed);
		do_clear = true;
	}
	if (y_key) {
		camera.updateApertureRadius(0.1f * speed);
		do_clear = true;
	}
	if (h_key) {
		camera.updateApertureRadius(-0.1f * speed);
		do_clear = true;
	}

	if (do_clear) {
		clearRender();
		do_clear = false;
	}

}

static void tick(int) {

	if (!paused) {

		cameraUpdate();
		render(camera);
		glutPostRedisplay();

	}

	glutTimerFunc(1, tick, 0);

}

static void displayTextPixel(int x, int y) {

	glRasterPos2f(2.0f * ((float) x / WIDTH) - 1.0f, 1.0f - 2.0f * ((float) y / HEIGHT));

}

static void displayTextLine(int l) {

	displayTextPixel(10, 20 + l * 14);
}

static void displayText(std::string s) {

	for (int i = 0; i < s.size(); i++) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, s[i]);
	}

}

static void displayUIText() {

	int render_count;
	double render_time;
	getRenderStatus(render_count, render_time);

	double samples_per_second = render_count / render_time;

	glPushMatrix();
	glLoadIdentity();

	displayTextLine(0);
	displayText("samples: ");
	displayText(std::to_string(render_count));

	displayTextLine(1);
	displayText("render time: ");
	displayText(std::to_string((int) render_time));
	displayText(" seconds");

	displayTextLine(2);
	displayText(std::to_string((int) samples_per_second));
	displayText(" samples per second");

	glPopMatrix();

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

	displayUIText();

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
	case 'c':
		clearRender();
		break;
	case 'w':
		w_key = true;
		break;
	case 'a':
		a_key = true;
		break;
	case 's':
		s_key = true;
		break;
	case 'd':
		d_key = true;
		break;
	case 'r':
		r_key = true;
		break;
	case 'f':
		f_key = true;
		break;
	case 't':
		t_key = true;
		break;
	case 'g':
		g_key = true;
		break;
	case 'y':
		y_key = true;
		break;
	case 'h':
		h_key = true;
		break;
	}

}

void keyboardUp(unsigned char key, int x, int y) {

	switch (key) {
	case 'w':
		w_key = false;
		break;
	case 'a':
		a_key = false;
		break;
	case 's':
		s_key = false;
		break;
	case 'd':
		d_key = false;
		break;
	case 'r':
		r_key = false;
		break;
	case 'f':
		f_key = false;
		break;
	case 't':
		t_key = false;
		break;
	case 'g':
		g_key = false;
		break;
	case 'y':
		y_key = false;
		break;
	case 'h':
		h_key = false;
		break;
	}

}

void mouse(int button, int state, int x, int y) {

	if (button == GLUT_LEFT_BUTTON) {
		if (state == GLUT_DOWN) {
			left_mouse = true;
			mouse_x = x;
			mouse_y = y;
			glutSetCursor(GLUT_CURSOR_NONE);
		} else {
			left_mouse = true;
			glutSetCursor(GLUT_CURSOR_INHERIT);
		}
	} else if (button == GLUT_RIGHT_BUTTON) {
		right_mouse = (state == GLUT_DOWN);
	}

}

void motion(int x, int y) {

	if (left_mouse && x != mouse_x && y != mouse_y) {

		int dx = x - mouse_x;
		int dy = y - mouse_y;

		camera.rotate(-TURN_SPEED * dx, -TURN_SPEED * dy);

		glutWarpPointer(mouse_x, mouse_y);

		do_clear = true;

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

	last_time = std::chrono::high_resolution_clock::now();

	glutIgnoreKeyRepeat(1);

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutKeyboardUpFunc(keyboardUp);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);

	tick(0);

	glutMainLoop();

	return 0;

}
