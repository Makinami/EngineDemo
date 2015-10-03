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

using namespace std;
using namespace DirectX;

class SkyClass
{
public:
	SkyClass();
	~SkyClass();

	int Init(ID3D11Device1* device, ID3D11DeviceContext1* mImmediateContext);

	void Draw(ID3D11DeviceContext1* mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight& light);

private:
	int Precompute(ID3D11DeviceContext1* mImmediateContext);

private:
	ID3D11ShaderResourceView* transmitanceSRV;
	ID3D11UnorderedAccessView* transmitanceUAV;

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
	ID3D11ComputeShader* copyinscatter1CS;
	ID3D11ComputeShader* inscatterSCS;
	ID3D11ComputeShader* irradianceNCS;
	ID3D11ComputeShader* inscatterNCS;
	ID3D11ComputeShader* copyIrradianceCS;
	ID3D11ComputeShader* copyInscatterNCS;

	struct cbIrradianceNType
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

	ID3D11Buffer* cbIrradianceN;
	ID3D11Buffer* cbPerFrameVS;
	ID3D11Buffer* cbPerFramePS;


	ID3D11Buffer* mScreenQuadVB;
	ID3D11Buffer* mScreenQuadIB;

	ID3D11InputLayout* mInputLayout;
	ID3D11VertexShader* mVertexShader;
	ID3D11PixelShader* mPixelShader;

	ID3D11RasterizerState* mRastStateBasic;

	struct Vertex
	{
		XMFLOAT3 Pos;
	};
};

