#pragma once

#pragma comment(lib,  "libnoise.lib")

#include <d3d11_1.h>

#pragma warning(disable:4838)
#include <DirectXMath.h>
#pragma warning(default:4838)

#define ReleaseCOM(x) { if(x){ x->Release(); x = nullptr; } }
#define SafeDelete(x) { delete x; x = nullptr; }

#include <memory>

#include "cameraclass.h"
#include "Lights.h"

class CloudsClass
{
public:
	CloudsClass();
	~CloudsClass();

	int Init(ID3D11Device1* device, ID3D11DeviceContext1* mImmediateContext);

	void Draw(ID3D11DeviceContext1* mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight& light);

public:
	int GenerateClouds(ID3D11DeviceContext1* mImmediateContext);

private:
	ID3D11ShaderResourceView* mCloudSRV;
	ID3D11UnorderedAccessView* mCloudUAV;

	ID3D11ComputeShader* mGenerateCS;

	ID3D11Buffer* mCloudsQuadVB;
	ID3D11Buffer* mCloudsQuadIB;

	ID3D11InputLayout* mInputLayout;
	ID3D11VertexShader* mVertexShader;
	ID3D11PixelShader* mPixelShader;

	struct cbPerFrameVSType
	{
		XMMATRIX gProjView;
		XMMATRIX gWorld;
	};

	struct cbPerFramePSType
	{
		XMFLOAT3 gCameraPos;
		XMFLOAT3 gSunDir;
		float x;
		float pad;
	};

	ID3D11Buffer* cbPerFrameVS;
	ID3D11Buffer* cbPerFramePS;

	ID3D11SamplerState* mSamplerStateTrilinear;
	ID3D11BlendState1* mBlendStateClouds;
};