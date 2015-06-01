#include "map.h"

MapClass::MapClass() :
	Terrain(nullptr),
	Water(nullptr)
{
}

MapClass::~MapClass()
{
}

bool MapClass::Init(ID3D11Device1* device, ID3D11DeviceContext1 * dc)
{
	Water = std::make_shared<WaterClass>();
	if (!Water->Init(device, dc))
	{
		LogError(L"Failed to initiate water");
		return false;
	}
	LogSuccess(L"Water initiated");



	Terrain = std::make_shared<TerrainClass>();
	Terrain->SetLogger(Logger);

	TerrainClass::InitInfo tii;
	tii.HeightMapFilename = L"Textures/terrain.raw";
	tii.LayerMapFilename0 = L"Textures/grass.dds";
	tii.LayerMapFilename1 = L"Textures/darkdirt.dds";
	tii.LayerMapFilename2 = L"Textures/stone.dds";
	tii.LayerMapFilename3 = L"Textures/lightdirt.dds";
	tii.LayerMapFilename4 = L"Textures/snow.dds";
	tii.BlendMapFilename = L"Textures/blend.dds";
	tii.HeightScale = 100.0f;
	tii.HeightmapWidth = 2049;
	tii.HeightmapHeight = 2049;
	tii.CellSpacing = 0.5f;

	if (!Terrain->Init(device, dc, tii))
	{
		LogError(L"Failed to initiate terrain");
		return false;
	}
	LogSuccess(L"Terrain initiated");

	return true;
}

void MapClass::Draw(ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera)
{
	Terrain->Draw(mImmediateContext, Camera);

	Water->Draw(mImmediateContext, Camera);
}

float MapClass::GetHeight(float x, float y)
{
	return Terrain->GetHeight(x, y);
}
