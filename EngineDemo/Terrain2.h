#pragma once

#define WIN32_LEAN_AND_MEAN

#include <d3d11_1.h>

#include <wrl\client.h>

#include "Utilities\Texture.h"

#include "loggerclass.h"
#include "Lights.h"
#include "cameraclass.h"
#include "CDLODQuadTree.h"

__declspec(align(16)) class TerrainClass2 : public HasLogger
{
public:
	TerrainClass2();
	~TerrainClass2();

	void* operator new(size_t i)
	{
		return _aligned_malloc(i, 16);
	}

	void operator delete(void* p)
	{
		_aligned_free(p);
	}

	bool Init(ID3D11Device1* device, ID3D11DeviceContext1* mImmediateContext);

	void Draw(ID3D11DeviceContext1* mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight& light);

private:
	HRESULT GenerateDistanceField(ID3D11Device1* device, ID3D11DeviceContext1* mImmediateContext);

	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> mHeighmapRawSRV;
	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> mProDF;

	std::unique_ptr<Texture> mOceanDFA;
	std::unique_ptr<Texture> mOceanDFB;

	std::unique_ptr<Texture> mHeighmap;

	ID3D11ComputeShader* mInitJFA;
	ID3D11ComputeShader* mStepJFA;
	ID3D11ComputeShader* mPostJFA;

	ID3D11ComputeShader* mProcessHM;

	struct {
		int step;
		float size[2];
		float mip;
	} JFAParams;

	Microsoft::WRL::ComPtr<ID3D11Buffer> mJFACB;

	CDLODFlatQuadTree terrainQuadTree;

	struct {
		XMMATRIX gWorldProj;
		XMFLOAT3 camPos;
		float pad;
	} MatrixBuffer;

	Microsoft::WRL::ComPtr<ID3D11Buffer> mMatrixCB;
	ID3D11PixelShader* mPixelShader;
	Microsoft::WRL::ComPtr<ID3D11VertexShader> mVertexShader;

	Microsoft::WRL::ComPtr<ID3D11InputLayout> mQuadIL;
	ID3D11VertexShader* mQuadVS;

	// TODO temp
	std::unique_ptr<Texture> mGenHeightmap;

	ID3D11ComputeShader* mCreateHM;
};

