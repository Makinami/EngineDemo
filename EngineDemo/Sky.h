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

#include "Utilities\Texture.h"

using namespace std;
using namespace DirectX;

class SkyClass : public Debug::HasPerformance
{
public:
	SkyClass();
	~SkyClass();

	int Init(ID3D11Device1* device, ID3D11DeviceContext1* mImmediateContext);

	void Draw(ID3D11DeviceContext1* mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight& light);
	
	void DrawToMap(ID3D11DeviceContext1* mImmediateContext, DirectionalLight& light);

	void DrawToScreen(ID3D11DeviceContext1* mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight& light);

	ID3D11ShaderResourceView* getTransmittanceSRV();

private:
	int Precompute(ID3D11DeviceContext1* mImmediateContext);

private:
	std::unique_ptr<Texture> newTransmittanceText;

	std::unique_ptr<Texture> newDeltaEText;

	std::vector<ID3D11ShaderResourceView*> deltaSSRV;
	std::vector<ID3D11UnorderedAccessView*> deltaSUAV;

	std::vector<std::unique_ptr<Texture>> newDeltaSText;

	std::unique_ptr<Texture> newIrradainceText;
	std::unique_ptr<Texture> newCopyIrradianceText;

	std::unique_ptr<Texture> newInscatterText;
	std::unique_ptr<Texture> newCopyInscatterText;

	std::unique_ptr<Texture> newDeltaJText;

	ID3D11ComputeShader* transmittanceCS;
	ID3D11ComputeShader* irradiance1CS;
	ID3D11ComputeShader* inscatter1CS;
	ID3D11ComputeShader* copyInscatter1CS;
	ID3D11ComputeShader* inscatterSCS;
	ID3D11ComputeShader* irradianceNCS;
	ID3D11ComputeShader* inscatterNCS;
	ID3D11ComputeShader* copyIrradianceCS;
	ID3D11ComputeShader* copyInscatterNCS;
	ID3D11ComputeShader* zeroIrradiance;

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

	D3D11_VIEWPORT skyMapViewport;

	std::unique_ptr<Texture> newMapText;
	
	UINT skyMapSize;
	
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