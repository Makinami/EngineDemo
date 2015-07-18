#pragma once

#if defined(DEBUG) || defined(_DEBUG)
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#endif
#include <memory>

#include "loggerclass.h"
#include "terrain.h"
#include "water.h"
#include "Water20.h"

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
class MapClass : public HasLogger
{
public:
	MapClass();
	~MapClass();

	bool Init(ID3D11Device1* device, ID3D11DeviceContext1* dc);

	void Shutdown();

	void Update(float dt, ID3D11DeviceContext1 * mImmediateContext);

	void Draw(ID3D11DeviceContext1* mImmediateContext, std::shared_ptr<CameraClass> Camera);
	void Draw20(ID3D11DeviceContext1* mImmediateContext, std::shared_ptr<CameraClass> Camera);

	void DrawDebug(ID3D11DeviceContext1* mImmediateContext);

	float GetHeight(float x, float y);

private:
	std::shared_ptr<TerrainClass> Terrain;
	std::shared_ptr<WaterClass> Water;
	std::shared_ptr<WaterClass20> Water20;

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
	};

	ID3D11Buffer* MatrixBuffer;
};