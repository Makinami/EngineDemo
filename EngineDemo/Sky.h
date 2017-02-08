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

#include "Utilities\Texture.h"

#include "MeshBuffer.h"

#include "PostFX.h"

class SkyClass
{
public:
	SkyClass();
	~SkyClass();

	int Init(ID3D11Device1* device, ID3D11DeviceContext1* mImmediateContext);

	HRESULT Shutdown();

	void Draw(ID3D11DeviceContext1* mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight& light);
	
	void DrawToMap(ID3D11DeviceContext1* mImmediateContext, DirectionalLight& light);

	void DrawToScreen(ID3D11DeviceContext1* mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight& light);

	void Process(ID3D11DeviceContext1* mImmediateContext, std::unique_ptr<PostFX::Canvas> const& Canvas, std::shared_ptr<CameraClass> Camera, DirectionalLight& light);

	void SetTransmittance(ID3D11DeviceContext1* mImmediateContext, int slot);

	ID3D11ShaderResourceView* getTransmittanceSRV();

private:
	HRESULT Precompute(ID3D11DeviceContext1* mImmediateContext);

private:
	std::unique_ptr<Texture> mTransmittanceTex;
	std::unique_ptr<Texture> mDeltaETex;
	std::unique_ptr<Texture> mIrradianceTex;
	std::unique_ptr<Texture> mIrradianceCopyTex;
	std::unique_ptr<Texture> mInscatterTex;
	std::unique_ptr<Texture> mInscatterCopyTex;
	std::unique_ptr<Texture> mDeltaJTex;

	std::vector<std::unique_ptr<Texture>> mDeltaSTex;
	std::vector<ID3D11ShaderResourceView*> deltaSSRV;
	std::vector<ID3D11UnorderedAccessView*> deltaSUAV;

	ID3D11ComputeShader* mTransmittanceCS;
	ID3D11ComputeShader* mIrradianceZeroCS;
	ID3D11ComputeShader* mIrradianceSingleCS;
	ID3D11ComputeShader* mIrradianceMultipleCS;
	ID3D11ComputeShader* mIrradianceAddCS;
	ID3D11ComputeShader* mInscatterSingleCS;
	ID3D11ComputeShader* mInscatterMultipleACS;
	ID3D11ComputeShader* mInscatterMultipleBCS;
	ID3D11ComputeShader* mInscatterCopyCS;
	ID3D11ComputeShader* mInscatterAddCS;

	struct nOrderType
	{
		int order[4];
	} nOrderParams;

	ID3D11Buffer* nOrderCB;

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
		float pad2;
		XMFLOAT4 gProj;
	} cbPerFramePSParams;

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

	MeshBuffer mScreenQuad;

	ID3D11InputLayout* mInputLayout;
	ID3D11VertexShader* mVertexShader;
	ID3D11PixelShader* mPixelShaderToCube;
	ID3D11PixelShader* mPixelShaderToScreen;
	ID3D11PixelShader* mPixelShaderPostFX;

	ID3D11VertexShader* mMapVertexShader;
	ID3D11PixelShader* mMapPixelShader;

	ID3D11SamplerState** mSamplerStateBasic; // 4 identical basic states

	struct Vertex
	{
		XMFLOAT3 Pos;
	};

	// TEMP
	ID3D11ShaderResourceView* transmittanceFile;
	ID3D11ShaderResourceView* inscatterFile;
	ID3D11ShaderResourceView* irradianceFile;
};