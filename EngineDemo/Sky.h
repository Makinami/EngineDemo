#pragma once

#include <d3d11_1.h>

#pragma warning(disable:4838)
#include <DirectXMath.h>
#pragma warning(default:4838)

#include <map>
#include <vector>

#define ReleaseCOM(x) { if(x){ x->Release(); x = nullptr; } }
#define SafeDelete(x) { delete x; x = nullptr; }

#include "terrain.h"
#include "cameraclass.h"

#include "Performance.h"

using namespace std;
using namespace DirectX;

class SkyClass : public Debug::HasPerformance
{
public:
	SkyClass();
	~SkyClass();

	int Init(ID3D11Device1* device, ID3D11DeviceContext1* mImmediateContext);

	void Draw(ID3D11DeviceContext1* mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight& light);

	void DrawToCube(ID3D11DeviceContext1* mImmediateContext, DirectionalLight& light);

	void DrawToMap(ID3D11DeviceContext1* mImmediateContext, DirectionalLight& light);

	void DrawToScreen(ID3D11DeviceContext1* mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight& light);

	ID3D11ShaderResourceView* getTransmittanceSRV();

private:
	int Precompute(ID3D11DeviceContext1* mImmediateContext);

private:
	ID3D11ShaderResourceView* transmitanceSRV;
	ID3D11UnorderedAccessView* transmitanceUAV;
	ID3D11Texture2D* transmittanceText;

	ID3D11ShaderResourceView* deltaESRV;
	ID3D11UnorderedAccessView* deltaEUAV;

	ID3D11ShaderResourceView** deltaSSRV;
	ID3D11UnorderedAccessView** deltaSUAV;

	ID3D11ShaderResourceView* irradianceSRV;
	ID3D11UnorderedAccessView* irradianceUAV;
	ID3D11Texture2D* irradianceText;
	ID3D11ShaderResourceView* copyIrradianceSRV;
	ID3D11UnorderedAccessView* copyIrradianceUAV;
	ID3D11Texture2D* copyIrradianceText;

	ID3D11ShaderResourceView* inscatterSRV;
	ID3D11UnorderedAccessView* inscatterUAV;
	ID3D11Texture3D* inscatterText;
	ID3D11ShaderResourceView* copyInscatterSRV;
	ID3D11UnorderedAccessView* copyInscatterUAV;
	ID3D11Texture3D* copyInscatterText;

	ID3D11ShaderResourceView* deltaJSRV;
	ID3D11UnorderedAccessView* deltaJUAV;

	ID3D11ComputeShader* transmittanceCS;
	ID3D11ComputeShader* irradiance1CS;
	ID3D11ComputeShader* inscatter1CS;
	ID3D11ComputeShader* copyInscatter1CS;
	ID3D11ComputeShader* inscatterSCS;
	ID3D11ComputeShader* irradianceNCS;
	ID3D11ComputeShader* inscatterNCS;
	ID3D11ComputeShader* copyIrradianceCS;
	ID3D11ComputeShader* copyInscatterNCS;

	int cos = D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT;

	struct cbNOrderType
	{
		int order[4];
	};

	struct cbPerFrameVSType
	{
		XMMATRIX gViewInverse;
		XMMATRIX gProjInverse;
	};

	struct cbPerFramePSType
	{
		XMFLOAT3 gCameraPos;
		float gExposure;
		XMFLOAT3 gSunDir;
		float pad;
	};

	struct skyMapBufferType
	{
		XMFLOAT3 sunDir;
		float pad;
	} skyMapParams;

	ID3D11Buffer* cbNOrder;
	ID3D11Buffer* cbPerFrameVS;
	ID3D11Buffer* cbPerFramePS;
	ID3D11Buffer* skyMapCB;

	const int CubeMapSize = 512;

	ID3D11RenderTargetView** cubeTextRTV; // 6 faces
	ID3D11ShaderResourceView* cubeTextSRV;
	ID3D11DepthStencilView* cubeMapDSV;
	D3D11_VIEWPORT cubeViewport;

	ID3D11RenderTargetView* mapRTV;
	ID3D11ShaderResourceView* mapSRV;
	
	UINT skyMapSize;

	std::shared_ptr<CameraClass> mCubeMapCamera[6];

	void BuildCubeFaceCamera(float x, float y, float z);

	ID3D11Buffer* mScreenQuadVB;
	ID3D11Buffer* mScreenQuadIB;

	ID3D11InputLayout* mInputLayout;
	ID3D11VertexShader* mVertexShader;
	ID3D11PixelShader* mPixelShaderToCube;
	ID3D11PixelShader* mPixelShaderToScreen;

	ID3D11VertexShader* mMapVertexShader;
	ID3D11PixelShader* mMapPixelShader;

	ID3D11RasterizerState* mRastStateBasic;
	ID3D11SamplerState** mSamplerStateBasic; // 4 identical basic states
	ID3D11SamplerState* mSamplerStateTrilinear;
	ID3D11SamplerState* mSamplerAnisotropic;
	ID3D11DepthStencilState* mDepthStencilStateSky;

	struct Vertex
	{
		XMFLOAT3 Pos;
	};

	// performace ids
	char drawSky;

	// TEMP
	ID3D11ShaderResourceView* transmittanceFile;
	ID3D11ShaderResourceView* inscatterFile;
	ID3D11ShaderResourceView* irradianceFile;
};