#include "common.h"
#include "geometry.h"
#include "scene.h"

#include "scenes.h"

//scene.addInstance(cube_model_index);
//scene.setDiffuse({0.6f, 0.6f, 0.6f});
//scene.scale(100.0f);
//scene.translate({0.0f, 110.0f, 0.0f});
//scene.setEmission({1.0f, 1.0f, 1.0f}, 3.0f);

//scene.addInstance(cube_model_index);
//scene.setDiffuse({0.0f, 1.0f, 0.0f});
//scene.scale(100.0f);
//scene.translate({105.0f, 55.0f, 0.0f});

//scene.addInstance(cube_model_index);
//scene.setDiffuse({1.0f, 0.0f, 0.0f});
//scene.scale(100.0f);
//scene.translate({-105.0f, 55.0f, 0.0f});

//scene.addInstance(cube_model_index);
//scene.setDiffuse({0.6f, 0.6f, 0.6f});
//scene.scale(100.0f);
//scene.translate({0.0f, 55.0f, -105.0f});

//scene.addInstance(cube_model_index);
//scene.setDiffuse({0.6f, 0.6f, 0.6f});
//scene.scale(100.0f);
//scene.translate({0.0f, 55.0f, 110.0f});




////scene.addSphere({0.0f, 3.0f, 0.0f}, 1.5f);
////scene.setDiffuse({0.7f, 0.4f, 1.0f});


//int model_index = scene.addModel("models/wt_teapot.obj");
//scene.addInstance(model_index);
//scene.setInterpolateNormals(true);
//scene.setDiffuse({0.7f, 0.4f, 1.0f});
//scene.setAttenuation({0.7f, 0.4f, 1.0f});
//scene.setTransmissionWeight(1.0f);
////scene.setScatter(3.0f);
////scene.setSpecularWeight(0.01f);
//scene.scale(2.5f);
//scene.translate({0.0f, 1.5f, 0.0f});

void buildScene(scene_t& scene, camera_t& camera, int width, int height) {

	camera.init({0.0f, 4.0f, 7.0f}, (float) width / height);
	camera.set_rotation(0.0f, -0.25f);
	scene.setMaxDepth(10);
	scene.setBackgroundEmission({0.4f, 0.6f, 0.8f});


	int cube_model_index = scene.addModel("models/cube.obj");

	scene.addInstance(cube_model_index);
	scene.setDiffuse({0.6f, 0.6f, 0.6f});
	scene.scale(100.0f);
	scene.translate({0.0f, -100.0f, 0.0f});

	scene.addInstance(cube_model_index);
	scene.setDiffuse({0.6f, 0.6f, 0.6f});
	scene.scale(100.0f);
	scene.translate({0.0f, 110.0f, 0.0f});
	scene.setEmission({1.0f, 1.0f, 1.0f}, 3.0f);

	scene.addInstance(cube_model_index);
	scene.setDiffuse({0.0f, 1.0f, 0.0f});
	scene.scale(100.0f);
	scene.translate({105.0f, 55.0f, 0.0f});

	scene.addInstance(cube_model_index);
	scene.setDiffuse({1.0f, 0.0f, 0.0f});
	scene.scale(100.0f);
	scene.translate({-105.0f, 55.0f, 0.0f});

	scene.addInstance(cube_model_index);
	scene.setDiffuse({0.6f, 0.6f, 0.6f});
	scene.scale(100.0f);
	scene.translate({0.0f, 55.0f, -105.0f});

	scene.addInstance(cube_model_index);
	scene.setDiffuse({0.6f, 0.6f, 0.6f});
	scene.scale(100.0f);
	scene.translate({0.0f, 55.0f, 110.0f});




	int model_index = scene.addModel("models/dragon.obj");
	scene.addInstance(model_index);
	scene.setInterpolateNormals(true);
	scene.setTransmissionWeight(1.0f);
	scene.setAttenuation({0.1f, 0.5f, 1.0f});
	scene.setScatter(1.0f);
	scene.scale(0.5f);
	scene.rotate_y(0.5f);

}
