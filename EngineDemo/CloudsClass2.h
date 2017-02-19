#pragma once

#pragma comment(lib,  "libnoise.lib")

#include <d3d11_1.h>

#pragma warning(disable:4838)
#include <DirectXMath.h>
#pragma warning(default:4838)

#define ReleaseCOM(x) { if(x){ x->Release(); x = nullptr; } }
#define SafeDelete(x) { delete x; x = nullptr; }

#include <memory>
#include <vector>
#include <string>
#include <wrl\client.h>

#include <unordered_map>
#include <array>

#include "cameraclass.h"
#include "Lights.h"

class CloudsClass2
{
public:
	CloudsClass2();
	~CloudsClass2();

	int Init(ID3D11Device1* device, ID3D11DeviceContext1* mImmediateContext);

	void Update(float dt);

	void Draw(ID3D11DeviceContext1* mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight& light, ID3D11ShaderResourceView* transmittanceSRV);
	
public:
	int GenerateClouds(ID3D11DeviceContext1* mImmediateContext);
	int GenerateCloudsParametrized(ID3D11DeviceContext1* mImmediateContext);

	HRESULT GenerateSeedGrad(ID3D11Device1* device, ID3D11DeviceContext1* mImmediateContext);

private:
	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> mCloudGeneralSRV;
	Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> mCloudGeneralUAV;

	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> mCloudDetailSRV;
	Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> mCloudDetailUAV;

	ID3D11ShaderResourceView* mCloudCurlSRV;
	ID3D11ShaderResourceView* mCloudTypesSRV;
	ID3D11ShaderResourceView* mWeatherSRV;

	Microsoft::WRL::ComPtr<ID3D11ComputeShader> mGenerateGenCS;
	Microsoft::WRL::ComPtr<ID3D11ComputeShader> mGenerateDetCS;
	Microsoft::WRL::ComPtr<ID3D11ComputeShader> mGenerateNoise;

	ID3D11Buffer* mScreenQuadVB;
	ID3D11Buffer* mScreenQuadIB;

	Microsoft::WRL::ComPtr<ID3D11InputLayout> mInputLayout;
	Microsoft::WRL::ComPtr<ID3D11VertexShader> mVertexShader;
	Microsoft::WRL::ComPtr<ID3D11PixelShader> mPixelShader;

	const int GEN_RES = 128;
	const int DET_RES = 32;
	const int FLW_RES = 128;

	struct cbPerFrameVSType
	{
		XMMATRIX gViewInverse;
		XMMATRIX gProjInverse;
	} cbPerFrameVSParams;

	struct cbPerFramePSType
	{
		XMFLOAT3 gCameraPos;
		float bExposure;
		XMFLOAT3 gSunDir;
		float time;
		std::array<XMFLOAT4, 10> parameters;
	} cbPerFramePSParams;

	std::array<std::array<char, 20>, 10> parametersNames;

	struct
	{
		XMINT4 gFrequency;
		int textSize;
		XMUINT3 pad;
	} cbGenerateNoiseParams;

	struct
	{
		XMINT4 baseFrequency;
		XMINT4 detailFrequency;
	} noiseFrequencies = { {1,1,1,1},{1,1,1,1} };

	ID3D11Buffer* cbPerFrameVS;
	ID3D11Buffer* cbPerFramePS;
	ID3D11Buffer* cbGenerataNoise;

	ID3D11SamplerState* mSamplerStateTrilinear;
	ID3D11SamplerState* mSamplerStateBilinearClamp;
	ID3D11BlendState1* mBlendStateClouds;
	ID3D11DepthStencilState* mDepthStencilState;

	Microsoft::WRL::ComPtr<ID3D11Texture3D> mRandomGrad;
	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> mRandomGradSRV;

	ID3D11Device1* dev;
};