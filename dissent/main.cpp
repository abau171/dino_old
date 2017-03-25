#include <iostream>

#include "GL/glew.h"
#include "GL/glut.h"

#include "common.h"
#include "geometry.h"
#include "render.h"

const int WIDTH = 640;
const int HEIGHT = 480;

unsigned char image_data[WIDTH][HEIGHT][3];

void updateImage(void) {

	if (!render((unsigned char*) image_data)) return;

}

void display() {

	updateImage();

	glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, image_data);
	glutSwapBuffers();
	glutPostRedisplay();

}

int main(int argc, char** argv) {

	if (!resetRender(WIDTH, HEIGHT)) return 1;

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(WIDTH, HEIGHT);
	glutCreateWindow("Dissent Path Tracer");

	glutDisplayFunc(display);

	glutMainLoop();

	return 0;

}
