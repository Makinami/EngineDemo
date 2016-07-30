#pragma once

#pragma comment(lib,  "libnoise.lib")

#include <d3d11_1.h>

#pragma warning(disable:4838)
#include <DirectXMath.h>
#pragma warning(default:4838)

#define ReleaseCOM(x) { if(x){ x->Release(); x = nullptr; } }
#define SafeDelete(x) { delete x; x = nullptr; }

#include <memory>
#include <wrl\client.h>

#include "cameraclass.h"
#include "Lights.h"

#include "TweakBar.h"

class CloudsClass2
{
public:
	CloudsClass2();
	~CloudsClass2();

	int Init(ID3D11Device1* device, ID3D11DeviceContext1* mImmediateContext);

	void Draw(ID3D11DeviceContext1* mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight& light, ID3D11ShaderResourceView* transmittanceSRV);

	void SetBar(std::shared_ptr<TweakBar> Bar);

public:
	int GenerateClouds(ID3D11DeviceContext1* mImmediateContext);

	HRESULT GenerateSeedGrad(ID3D11Device1* device, ID3D11DeviceContext1* mImmediateContext);

private:
	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> mCloudGeneralSRV;
	Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> mCloudGeneralUAV;

	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> mCloudDetailSRV;
	Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> mCloudDetailUAV;

	ID3D11ShaderResourceView* mCloudCurlSRV;
	ID3D11ShaderResourceView* mCloudTypesSRV;
	ID3D11ShaderResourceView* mWeatherSRV;

	ID3D11ComputeShader* mGenerateGenCS;
	ID3D11ComputeShader* mGenerateDetCS;

	ID3D11Buffer* mScreenQuadVB;
	ID3D11Buffer* mScreenQuadIB;

	ID3D11InputLayout* mInputLayout;
	ID3D11VertexShader* mVertexShader;
	ID3D11PixelShader* mPixelShader;

	const int GEN_RES = 128;
	const int DET_RES = 32;
	const int FLW_RES = 128;

	struct cbPerFrameVSType
	{
		XMMATRIX gViewInverse;
		XMMATRIX gProjInverse;
	};

	struct cbPerFramePSType
	{
		XMFLOAT3 gCameraPos;
		float bExposure;
		XMFLOAT3 gSunDir;
		float pad;
	} cbPerFramePSParams;

	ID3D11Buffer* cbPerFrameVS;
	ID3D11Buffer* cbPerFramePS;

	ID3D11SamplerState* mSamplerStateTrilinear;
	ID3D11BlendState1* mBlendStateClouds;
	ID3D11DepthStencilState* mDepthStencilState;

	Microsoft::WRL::ComPtr<ID3D11Texture3D> mRandomGrad;
	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> mRandomGradSRV;
};