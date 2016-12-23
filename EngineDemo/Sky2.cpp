#include "Sky2.h"

#include "Shaders\Sky2\Structures.hlsli"
#include "Utilities\CreateShader.h"
#include "Utilities\CreateBuffer.h"
#include "Utilities\MapResources.h"

using namespace std;

SkyClass2::SkyClass2() :
	mTransmittanceCS(nullptr)
{
}

SkyClass2::~SkyClass2()
{
	Shutdown();
}

HRESULT SkyClass2::Init(ID3D11Device1 * device, ID3D11DeviceContext1 * mImmediateContext)
{
	/*
		2D
	*/
	// transmittance
	EXIT_ON_NULL(mTransmittanceTex =
				 TextureFactory::CreateTexture(D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
											   DXGI_FORMAT_R32G32B32A32_FLOAT,
											   TRANSMITTANCE_W, TRANSMITTANCE_H));

	// deltaE
	EXIT_ON_NULL(mDeltaETex =
				 TextureFactory::CreateTexture(D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
											   DXGI_FORMAT_R32G32B32A32_FLOAT,
											   SKY_W, SKY_H));

	// irradiance
	EXIT_ON_NULL(mIrradianceTex =
				 TextureFactory::CreateTexture(D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
											   DXGI_FORMAT_R32G32B32A32_FLOAT,
											   SKY_W, SKY_H));

	EXIT_ON_NULL(mIrradianceCopyTex =
				 TextureFactory::CreateTexture(D3D11_BIND_UNORDERED_ACCESS,
											   DXGI_FORMAT_R32G32B32A32_FLOAT,
											   SKY_W, SKY_H));

	/*
		3D
	*/
	D3D11_TEXTURE3D_DESC text3Desc;
	text3Desc.Width = RES_SZ*RES_VS;
	text3Desc.Height = RES_VZ;
	text3Desc.Depth = RES_ALT;
	text3Desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	text3Desc.CPUAccessFlags = 0;
	text3Desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	text3Desc.MipLevels = 1;
	text3Desc.MiscFlags = 0;
	text3Desc.Usage = D3D11_USAGE_DEFAULT;

	// deltaS
	mDeltaSTex.resize(2);
	EXIT_ON_NULL(mDeltaSTex[0] =
				 TextureFactory::CreateTexture(text3Desc));
	EXIT_ON_NULL(mDeltaSTex[1] =
				 TextureFactory::CreateTexture(text3Desc));

	deltaSSRV.resize(2);
	deltaSSRV[0] = mDeltaSTex[0]->GetSRV();
	deltaSSRV[1] = mDeltaSTex[1]->GetSRV();

	deltaSUAV.resize(2);
	deltaSUAV[0] = mDeltaSTex[0]->GetUAV();
	deltaSUAV[1] = mDeltaSTex[1]->GetUAV();

	// inscatter
	EXIT_ON_NULL(mInscatterTex =
				 TextureFactory::CreateTexture(text3Desc));

	EXIT_ON_NULL(mInscatterCopyTex =
				 TextureFactory::CreateTexture(text3Desc));

	// deltaJ
	EXIT_ON_NULL(mDeltaJTex =
				 TextureFactory::CreateTexture(text3Desc));

	// compute shaders
	if (!CreateCSFromFile(L"..\\Debug\\Shaders\\Sky2\\transmittance.cso", device, mTransmittanceCS) ||
		!CreateCSFromFile(L"..\\Debug\\Shaders\\Sky2\\irradianceSingle.cso", device, mIrradianceSingleCS) ||
		!CreateCSFromFile(L"..\\Debug\\Shaders\\Sky2\\inscatterSingle.cso", device, mInscatterSingleCS) ||
		!CreateCSFromFile(L"..\\Debug\\Shaders\\Sky2\\irradianceZero.cso", device, mIrradianceZeroCS) ||
		!CreateCSFromFile(L"..\\Debug\\Shaders\\Sky2\\inscatterCopy.cso", device, mInscatterCopyCS) ||
		!CreateCSFromFile(L"..\\Debug\\Shaders\\Sky2\\inscatterMultipleA.cso", device, mInscatterMultipleACS) ||
		!CreateCSFromFile(L"..\\Debug\\Shaders\\Sky2\\irradianceMultiple.cso", device, mIrradianceMultipleCS) ||
		!CreateCSFromFile(L"..\\Debug\\Shaders\\Sky2\\inscatterMultipleB.cso", device, mInscatterMultipleBCS) ||
		!CreateCSFromFile(L"..\\Debug\\Shaders\\Sky2\\irradianceAdd.cso", device, mIrradianceAddCS) ||
		!CreateCSFromFile(L"..\\Debug\\Shaders\\Sky2\\inscatterAdd.cso", device, mInscatterAddCS))
	{
		Shutdown();
		return E_FAIL | E_ABORT;
	}

	CreateConstantBuffer(device, sizeof(nOrderType), nOrderCB, "nOrderCB");

	//Precompute(mImmediateContext);
	
	return S_OK;
}

HRESULT SkyClass2::Shutdown()
{
	ReleaseCOM(mTransmittanceCS);
	ReleaseCOM(mIrradianceSingleCS);
	ReleaseCOM(mInscatterSingleCS);
	ReleaseCOM(mInscatterCopyCS);
	ReleaseCOM(mIrradianceZeroCS);

	mTransmittanceTex.release();
	mDeltaETex.release();
	mInscatterTex.release();

	return S_OK;
}

void SkyClass2::Draw(ID3D11DeviceContext1 * mImmediateContext)
{
	//Precompute(mImmediateContext);
	mImmediateContext->PSSetShaderResources(23, 1, mTransmittanceTex->GetAddressOfSRV());
	mImmediateContext->PSSetShaderResources(24, 1, mDeltaETex->GetAddressOfSRV());
	mImmediateContext->PSSetShaderResources(25, 2, &deltaSSRV[0]);
	mImmediateContext->PSSetShaderResources(27, 1, mIrradianceTex->GetAddressOfSRV());
	mImmediateContext->PSSetShaderResources(28, 1, mInscatterTex->GetAddressOfSRV());
	mImmediateContext->PSSetShaderResources(29, 1, mDeltaJTex->GetAddressOfSRV());
}

//void SkyClass2::Process(ID3D11DeviceContext1 * mImmediateContext, std::unique_ptr<PostFX::Canvas> const & Canvas, std::shared_ptr<CameraClass> Camera, DirectionalLight & light)
//{
//}

HRESULT SkyClass2::Precompute(ID3D11DeviceContext1 * mImmediateContext)
{
	ID3D11UnorderedAccessView* ppUAViewNULL[2] = { NULL, NULL };
	ID3D11ShaderResourceView* ppSRVNULL[4] = { NULL, NULL, NULL, NULL };

	// line 1
	// T(x,v) = T(x,x0(x,v))
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, mTransmittanceTex->GetAddressOfUAV(), nullptr);
	mImmediateContext->CSSetShader(mTransmittanceCS, nullptr, 0);

	mImmediateContext->Dispatch(TRANSMITTANCE_W / 16, TRANSMITTANCE_H / 16, 1);

	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);

	// line 2
	// deltaE = eps[L](x,s)
	mImmediateContext->CSSetShaderResources(0, 1, mTransmittanceTex->GetAddressOfSRV());
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, mDeltaETex->GetAddressOfUAV(), nullptr);
	mImmediateContext->CSSetShader(mIrradianceSingleCS, nullptr, 0);

	mImmediateContext->Dispatch(SKY_W / 16, SKY_H / 16, 1);

	mImmediateContext->CSSetShaderResources(0, 1, ppSRVNULL);
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);

	// line 3
	// deltaS = S[L](x,v,s)
	mImmediateContext->CSSetShaderResources(0, 1, mTransmittanceTex->GetAddressOfSRV());
	mImmediateContext->CSSetUnorderedAccessViews(0, 2, &deltaSUAV[0], nullptr);
	mImmediateContext->CSSetShader(mInscatterSingleCS, nullptr, 0);

	mImmediateContext->Dispatch(RES_SZ*RES_VS / 16, RES_VZ / 16, RES_ALT);

	mImmediateContext->CSSetShaderResources(0, 1, ppSRVNULL);
	mImmediateContext->CSSetUnorderedAccessViews(0, 2, ppUAViewNULL, nullptr);

	// line 4
	// E = 0
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, mIrradianceTex->GetAddressOfUAV(), nullptr);
	mImmediateContext->CSSetShader(mIrradianceZeroCS, nullptr, 0);

	mImmediateContext->Dispatch(SKY_W / 16, SKY_H / 16, 1);

	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);

	// line 5
	// S = deltaS
	mImmediateContext->CSSetShaderResources(0, 2, &deltaSSRV[0]);
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, mInscatterTex->GetAddressOfUAV(), nullptr);
	mImmediateContext->CSSetShader(mInscatterCopyCS, nullptr, 0);

	mImmediateContext->Dispatch(RES_SZ*RES_VS / 16, RES_VZ / 16, RES_ALT);

	mImmediateContext->CSSetShaderResources(0, 2, ppSRVNULL);
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);

	// line 6
	// for loop
	for (int order = 2; order <= 2; ++order)
	{
		nOrderParams.order[0] = order;
		MapResources(mImmediateContext, nOrderCB, nOrderParams);
		mImmediateContext->CSSetConstantBuffers(0, 1, &nOrderCB);

		// line 7
		// deltaJ = J[T*alpha/PI*deltaE + deltaS](x,v,s)
		mImmediateContext->CSSetShaderResources(0, 1, mTransmittanceTex->GetAddressOfSRV());
		mImmediateContext->CSSetShaderResources(1, 1, mIrradianceTex->GetAddressOfSRV());
		mImmediateContext->CSSetShaderResources(2, 2, &deltaSSRV[0]);
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, mDeltaJTex->GetAddressOfUAV(), nullptr);
		mImmediateContext->CSSetShader(mInscatterMultipleACS, nullptr, 0);

		mImmediateContext->Dispatch(RES_SZ*RES_VS / 16, RES_VZ / 16, RES_ALT);

		mImmediateContext->CSSetShaderResources(0, 4, ppSRVNULL);
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);

		// line 8
		// deltaE = eps[T*alpha/PI*deltaE + deltaS](x,s) = eps[deltaS](x,s)
		mImmediateContext->CSSetShaderResources(0, 2, &deltaSSRV[0]);
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, mDeltaETex->GetAddressOfUAV(), nullptr);
		mImmediateContext->CSSetShader(mIrradianceMultipleCS, nullptr, 0);

		mImmediateContext->Dispatch(SKY_W / 16, SKY_H / 16, 1);

		mImmediateContext->CSSetShaderResources(0, 2, ppSRVNULL);
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);

		// line 9
		// deltaS = integral T*deltaJ
		mImmediateContext->CSSetShaderResources(0, 1, mTransmittanceTex->GetAddressOfSRV());
		mImmediateContext->CSSetShaderResources(1, 1, mDeltaJTex->GetAddressOfSRV());
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, mInscatterTex->GetAddressOfUAV(), nullptr);
		mImmediateContext->CSSetShader(mInscatterMultipleBCS, nullptr, 0);

		mImmediateContext->Dispatch(RES_SZ*RES_VS / 16, RES_VZ / 16, RES_ALT);

		mImmediateContext->CSSetShaderResources(0, 1, ppSRVNULL);
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);

		// line 10
		// E = E + deltaE
		mImmediateContext->CSSetShaderResources(0, 1, mDeltaETex->GetAddressOfSRV());
		mImmediateContext->CSSetShaderResources(1, 1, mIrradianceTex->GetAddressOfSRV());
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, mIrradianceCopyTex->GetAddressOfUAV(), nullptr);
		mImmediateContext->CSSetShader(mIrradianceAddCS, nullptr, 0);

		mImmediateContext->Dispatch(SKY_W / 16, SKY_H / 16, 1);

		mImmediateContext->CSSetShaderResources(0, 1, ppSRVNULL);
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);

		mImmediateContext->CopyResource(mIrradianceTex->GetTexture(), mIrradianceCopyTex->GetTexture());

		// line 11
		// S = S + deltaS
		mImmediateContext->CSSetShaderResources(0, 1, mInscatterTex->GetAddressOfSRV());
		mImmediateContext->CSSetShaderResources(1, 1, mDeltaSTex[0]->GetAddressOfSRV());
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, mInscatterCopyTex->GetAddressOfUAV(), nullptr);
		mImmediateContext->CSSetShader(mInscatterAddCS, nullptr, 0);

		mImmediateContext->Dispatch(RES_SZ*RES_VS / 16, RES_VZ / 16, RES_ALT);

		mImmediateContext->CSSetShaderResources(0, 1, ppSRVNULL);
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);

		mImmediateContext->CopyResource(mInscatterTex->GetTexture(), mInscatterCopyTex->GetTexture());
	}

	return E_NOTIMPL;
}
