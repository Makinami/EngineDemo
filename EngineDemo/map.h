#pragma once

#if defined(DEBUG) || defined(_DEBUG)
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#endif
#include <memory>

#include "loggerclass.h"
#include "terrain.h"
#include "water.h"

class MapClass : public HasLogger
{
public:
	MapClass();
	~MapClass();

	bool Init(ID3D11Device1* device, ID3D11DeviceContext1* dc);

	void Shutdown();

	void Draw(ID3D11DeviceContext1* mImmediateContext, std::shared_ptr<CameraClass> Camera);

	float GetHeight(float x, float y);

private:
	std::shared_ptr<TerrainClass> Terrain;
	std::shared_ptr<WaterClass> Water;
};