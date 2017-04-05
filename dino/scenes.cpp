#include "common.h"
#include "geometry.h"
#include "scene.h"

#include "scenes.h"

static void buildViScene(scene_t& scene, camera_t& camera, int width, int height) {

	scene.setMaxDepth(4);
	scene.setBackgroundEmission({0.1f, 0.2f, 0.4f});

	camera.init({0.0f, 4.0f, 7.0f}, (float) width / height);
	camera.set_rotation(0.0f, -0.25f);

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

}

static void buildSkullScene(scene_t& scene, camera_t& camera, int width, int height) {

	scene.setMaxDepth(4);
	scene.setBackgroundEmission({0.4f, 0.6f, 0.9f});

	camera.init({-5.6f, 4.2f, 4.4f}, (float) width / height);
	camera.set_rotation(-1.0f, -0.3f);

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

}

static void buildCarScene(scene_t& scene, camera_t& camera, int width, int height) {

	scene.setMaxDepth(4);
	scene.setBackgroundEmission({0.4f, 0.6f, 0.9f});

	camera.init({0.0f, 4.0f, 7.0f}, (float) width / height);
	camera.set_rotation(0.0f, -0.25f);

	int car_model_index = scene.addModel("car.obj");
	scene.addInstance(car_model_index);
	scene.setDiffuse({1.0f, 0.0f, 0.0f});
	scene.setSpecularWeight(0.05f);
	scene.setSpecularPower(10000.0f);
	scene.setInterpolateNormals(true);
	scene.scale(0.01f);
	scene.translate({0.0f, 0.975f, 0.0f});
	scene.rotate_y(-0.8f);

	scene.addSphere({0.0f, -999.0f, 0.0f}, 1000.0f);
	scene.setDiffuse({0.2f, 0.4f, 0.1f});

	scene.addSphere({-50.0f, 80.0f, 0.0f}, 20.0f);
	scene.setEmission({1.0f, 0.9f, 0.8f}, 32.0f);

}

static void buildTeapotScene(scene_t& scene, camera_t& camera, int width, int height) {

	scene.setMaxDepth(10);

	scene.setBackgroundEmission({0.4f, 0.6f, 0.9f});

	camera.init({0.0f, 4.0f, 7.0f}, (float) width / height);
	camera.set_rotation(0.0f, -0.25f);

	int teapot_model_index = scene.addModel("teapot.obj");

	scene.addInstance(teapot_model_index);
	scene.setDiffuse({0.8f, 0.5f, 0.0f});
	scene.setInterpolateNormals(true);
	scene.scale(0.02f);
	scene.rotate_y(-0.8f);
	scene.translate({0.7f, 0.8f, 0.0f});

	scene.addSphere({0.0f, 4.0f, 0.0f}, 1.0f);
	scene.setEmission({1.0f, 1.0f, 1.0f}, 10.0f);

	scene.addInstance(teapot_model_index);
	scene.setInterpolateNormals(true);
	scene.setSpecularWeight(1.0f);
	scene.setSpecular({1.0f, 0.5f, 0.0f});
	scene.setSpecularPower(100.0f);
	scene.scale(0.01f);
	scene.rotate_y(3.9416f);
	scene.translate({-1.2f, 0.8f, 0.0f});

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

}

static void buildCornellBoxScene(scene_t& scene, camera_t& camera, int width, int height) {

	scene.setMaxDepth(10);

	camera.init({0.0f, 0.0f, 5.0f}, (float) width / height);
	camera.set_rotation(0.0f, 0.0f);

	// red wall
	scene.addSphere({-1002.0f, 0.0f, 0.0f}, 1000.0f);
	scene.setDiffuse({1.0f, 0.0f, 0.0f});

	// green wall
	scene.addSphere({1002.0f, 0.0f, 0.0f}, 1000.0f);
	scene.setDiffuse({0.0f, 1.0f, 0.0f});

	// top wall
	scene.addSphere({0.0f, 1002.0f, 0.0f}, 1000.0f);
	scene.setDiffuse({0.5f, 0.5f, 0.5f});

	// bottom wall
	scene.addSphere({0.0f, -1002.0f, 0.0f}, 1000.0f);
	scene.setDiffuse({0.5f, 0.5f, 0.5f});

	// back wall
	scene.addSphere({0.0f, 0.0f, -1002.0f}, 1000.0f);
	scene.setDiffuse({0.5f, 0.5f, 0.5f});

	// ceiling light
	scene.addSphere({0.0f, 101.995f, 0.0f}, 100.0f);
	scene.setEmission({1.0f, 1.0f, 1.0f}, 10.0f);

	// yellow light
	scene.addSphere({-1.5f, 0.0f, -1.5f}, 0.3f);
	scene.setEmission({1.0f, 0.5f, 0.0f}, 50.0f);

	// diffuse ball
	scene.addSphere({0.0f, -1.5f, 0.0f}, 0.5f);
	scene.setDiffuse({0.5f, 0.5f, 0.5f});

	// reflective ball
	scene.addSphere({1.3f, -1.3f, -1.3f}, 0.7f);
	scene.setSpecularWeight(1.0f);

	// glass ball
	scene.addSphere({-1.0f, -1.3f, -1.0f}, 0.5f);
	scene.setTransmissionWeight(1.0f);
	scene.setRefractiveIndex(1.8f);

	// glossy ball
	scene.addSphere({0.0f, 0.0f, -1.3f}, 0.7f);
	scene.setDiffuse({0.0f, 0.0f, 1.0f});
	scene.setSpecularWeight(0.1f);
	scene.setSpecularPower(1000.0f);

}

void buildScene(scene_t& scene, camera_t& camera, int width, int height) {

	//buildViScene(scene, camera, width, height);
	//buildSkullScene(scene, camera, width, height);
	//buildCarScene(scene, camera, width, height);
	//buildTeapotScene(scene, camera, width, height);
	buildCornellBoxScene(scene, camera, width, height);

}
