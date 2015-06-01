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
#include "map.h"

class PlayerClass : public HasLogger
{
public:
	PlayerClass::PlayerClass();
	PlayerClass::~PlayerClass();

	void Init();

	void SetMap(std::shared_ptr<MapClass> iMap);
	void SetCamera(std::shared_ptr<CameraClass> iCamera);
	void SetInput(std::shared_ptr<InputClass> iInput);

	void React(float dt);

private:
	XMFLOAT3 mPosition;

	const float WalkV = 2.0f;
	const float RunV = 50.0f;

	std::shared_ptr<CameraClass> Camera;
	std::shared_ptr<InputClass> Input;
	std::shared_ptr<MapClass> Map;
};