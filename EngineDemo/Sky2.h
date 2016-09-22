#pragma once

#include <d3d11_1.h>

#pragma warning(disable:4838)
#include <DirectXMath.h>
#pragma warning(default:4838)

#include <map>
#include <vector>

#define ReleaseCOM(x) { if(x){ x->Release(); x = nullptr; } }
#define SafeDelete(x) { delete x; x = nullptr; }

#include "Utilities\Texture.h"

class SkyClass2
{
public:
	SkyClass2();
	~SkyClass2();

	HRESULT Init(ID3D11Device1* device, ID3D11DeviceContext1* mImmediateContext);

	HRESULT Shutdown();

	void Draw(ID3D11DeviceContext1* mImmediateContext);

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

	// TEMP (?)
	struct nOrderType
	{
		int order[4];
	} nOrderParams;

	ID3D11Buffer* nOrderCB;
};

