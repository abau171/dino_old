#include <iostream>

#include "GL/glew.h"
#include "GL/glut.h"

const int WIDTH = 640;
const int HEIGHT = 480;

unsigned char image_data[WIDTH][HEIGHT][3];

void updateImage(void) {

	for (int x = 0; x < WIDTH; x++) {
		for (int y = 0; y < HEIGHT; y++) {
			image_data[x][y][0] = (x + y) % 256;
			image_data[x][y][1] = (x + y) % 256;
			image_data[x][y][2] = (x + y) % 256;
		}
	}

}

void display() {

	updateImage();

	glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, image_data);
	glutSwapBuffers();
	glutPostRedisplay();

}

int main(int argc, char** argv) {

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(WIDTH, HEIGHT);
	glutCreateWindow("Dissent Path Tracer");

	glutDisplayFunc(display);

	glutMainLoop();

	return 0;

}
