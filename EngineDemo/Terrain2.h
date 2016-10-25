#pragma once

#define WIN32_LEAN_AND_MEAN

#include <d3d11_1.h>

#include <wrl\client.h>

#include "Utilities\Texture.h"

#include "loggerclass.h"

class TerrainClass2 : public HasLogger
{
public:
	TerrainClass2();
	~TerrainClass2();

	bool Init(ID3D11Device1* device, ID3D11DeviceContext1* mImmediateContext);

private:
	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> mHeighmapRawSRV;

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
};

