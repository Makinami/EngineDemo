#pragma once

#if defined(DEBUG) || defined(_DEBUG)
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#endif
#include <memory>

#include "WaterBruneton.h"

#include "loggerclass.h"
#include "Performance.h"
#include "terrain.h"
#include "Water.h"
#include "Sky.h"
#include "CloudsClass.h"
#include "CloudsClass2.h"
#include "Ocean.h"

#include "shadowmapclass.h"

/*struct DirectionalLight
{
	DirectionalLight() { ZeroMemory(this, sizeof(this)); }

	XMFLOAT4 Ambient;
	XMFLOAT4 Diffuse;
	XMFLOAT4 Specular;
	XMFLOAT3 Direction;
	float Pad; // Pad the last float so we can set an array of lights if we wanted.
};
*/
class MapClass : public HasLogger, public Debug::HasPerformance
{
public:
	MapClass();
	~MapClass();

	bool Init(ID3D11Device1* device, ID3D11DeviceContext1* dc);
	
	void Shutdown();

	void Update(float dt, ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera);

	void Draw(ID3D11DeviceContext1* mImmediateContext, std::shared_ptr<CameraClass> Camera);
	void Draw20(ID3D11DeviceContext1* mImmediateContext, std::shared_ptr<CameraClass> Camera);

	void DrawDebug(ID3D11DeviceContext1* mImmediateContext, std::shared_ptr<CameraClass> Camera);

	float GetHeight(float x, float y);

private:
	std::shared_ptr<TerrainClass> Terrain;
	std::shared_ptr<WaterClass> Water;
	std::shared_ptr<SkyClass> Sky;
	std::shared_ptr<CloudsClass> Clouds;
	std::shared_ptr<CloudsClass2> Clouds2;
	std::unique_ptr<WaterBruneton> WaterB;
	std::unique_ptr<OceanClass> Ocean;

	// temp
	std::unique_ptr<ShadowMapClass> ShadowMap;
	DirectionalLight light;

	// TEMP
	ID3D11Buffer* mScreenQuadVB;
	ID3D11Buffer* mScreenQuadIB;

	ID3D11InputLayout* mDebugIL;
	ID3D11VertexShader* mDebugVS;
	ID3D11PixelShader* mDebugPS;

	struct MatrixBufferType
	{
		XMMATRIX gWorldProj;
	} MatrixBufferParams;

	ID3D11Buffer* MatrixBuffer;

	// CUBE
	ID3D11Buffer* mCubeVB;
	ID3D11Buffer* mCubeIB;

	ID3D11InputLayout* mCubeIL;
	ID3D11VertexShader* mCubeVS;
	ID3D11PixelShader* mCubePS;
};