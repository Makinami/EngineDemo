#pragma once

#define CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

#include <memory>

#pragma warning(disable:4838)
#include <DirectXMath.h>
#pragma warning(default:4838)

#include "cameraclass.h"
#include "inputclass.h"
#include "terrain.h"

class PlayerClass
{
public:
	PlayerClass::PlayerClass();
	PlayerClass::~PlayerClass();

	void Init();

	void SetTerrain(std::shared_ptr<TerrainClass> iMap);
	void SetCamera(std::shared_ptr<CameraClass> iCamera);
	void SetInput(std::shared_ptr<InputClass> iInput);

	void React(float dt);

private:
	XMFLOAT3 mPosition;

	float WalkV = 2.0f;
	float RunV = 5.0f;

	std::shared_ptr<CameraClass> Camera;
	std::shared_ptr<InputClass> Input;
	std::shared_ptr<TerrainClass> Terrain; //temp: later map
};