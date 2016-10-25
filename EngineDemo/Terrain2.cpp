#include "Terrain2.h"

#include "ShadersManager.h"
#include "DDSTextureLoader.h"
#include "ScreenGrab.h"
#include "Utilities\CreateBuffer.h"
#include "Utilities\MapResources.h"
using namespace DirectX;

TerrainClass2::TerrainClass2()
{
}


TerrainClass2::~TerrainClass2()
{
}

bool TerrainClass2::Init(ID3D11Device1 * device, ID3D11DeviceContext1 * mImmediateContext)
{
	mInitJFA = ShadersManager::Instance()->GetCS("Terrain2::initJFA");
	mStepJFA = ShadersManager::Instance()->GetCS("Terrain2::stepJFA");
	mPostJFA = ShadersManager::Instance()->GetCS("Terrain2::postJFA");

	mProcessHM = ShadersManager::Instance()->GetCS("Terrain2::procesHeighmap");

	CreateDDSTextureFromFile(device, L"terrain.dds", nullptr, &mHeighmapRawSRV);

	EXIT_ON_NULL(mOceanDFA = 
				 TextureFactory::CreateTexture(D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
											   DXGI_FORMAT_R32G32B32A32_FLOAT,
											   256, 256));

	EXIT_ON_NULL(mOceanDFB =
				 TextureFactory::CreateTexture(D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
											   DXGI_FORMAT_R32G32B32A32_FLOAT,
											   256, 256));

	EXIT_ON_NULL(mHeighmap =
				 TextureFactory::CreateTexture(D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
											   DXGI_FORMAT_R16_FLOAT,
											   4096, 4096));

	CreateConstantBuffer(device, sizeof(JFAParams), mJFACB);

	ID3D11UnorderedAccessView* ppUAViewNULL[2] = { NULL, NULL };
	ID3D11ShaderResourceView* ppSRVNULL[4] = { NULL, NULL, NULL, NULL };

	JFAParams.size[0] = 256;
	JFAParams.size[1] = 256;
	JFAParams.mip = log2(4096 / 256);
	MapResources(mImmediateContext, mJFACB.Get(), JFAParams);
	mImmediateContext->CSSetConstantBuffers(0, 1, mJFACB.GetAddressOf());

	mImmediateContext->CSSetShaderResources(0, 1, mHeighmapRawSRV.GetAddressOf());
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, mOceanDFA->GetAddressOfUAV(), nullptr);
	mImmediateContext->CSSetShader(mInitJFA, nullptr, 0);

	mImmediateContext->Dispatch(256 / 16, 256 / 16, 1);

	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);

	mImmediateContext->CSSetShader(mStepJFA, nullptr, 0);
	
	for (int step = 128; step; step /= 2)
	{
		JFAParams.step = step;
		MapResources(mImmediateContext, mJFACB.Get(), JFAParams);
		mImmediateContext->CSSetConstantBuffers(0, 1, mJFACB.GetAddressOf());

		mImmediateContext->CSSetUnorderedAccessViews(0, 1, mOceanDFA->GetAddressOfUAV(), 0);
		mImmediateContext->CSSetUnorderedAccessViews(1, 1, mOceanDFB->GetAddressOfUAV(), 0);

		mImmediateContext->Dispatch(256 / 16, 256 / 16, 1);

		mImmediateContext->CSSetUnorderedAccessViews(0, 2, ppUAViewNULL, 0);

		std::swap(mOceanDFA, mOceanDFB);
	}
	
	JFAParams.mip = 4096.0f / 256.0f;
	MapResources(mImmediateContext, mJFACB.Get(), JFAParams);
	mImmediateContext->CSSetConstantBuffers(0, 1, mJFACB.GetAddressOf());

	mImmediateContext->CSSetUnorderedAccessViews(0, 1, mOceanDFA->GetAddressOfUAV(), 0);
	mImmediateContext->CSSetUnorderedAccessViews(1, 1, mOceanDFB->GetAddressOfUAV(), 0);
	mImmediateContext->CSSetShader(mPostJFA, nullptr, 0);

	mImmediateContext->Dispatch(256 / 16, 256 / 16, 1);

	mImmediateContext->CSSetUnorderedAccessViews(0, 2, ppUAViewNULL, nullptr);

	std::swap(mOceanDFA, mOceanDFB);

	mImmediateContext->CSSetShaderResources(0, 1, mHeighmapRawSRV.GetAddressOf());
	mImmediateContext->CSSetShaderResources(1, 1, mOceanDFA->GetAddressOfSRV());
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, mHeighmap->GetAddressOfUAV(), nullptr);
	mImmediateContext->CSSetShader(mProcessHM, nullptr, 0);

	mImmediateContext->Dispatch(4096 / 16, 4096 / 16, 1);

	mImmediateContext->CSSetUnorderedAccessViews(0, 2, ppUAViewNULL, nullptr);

	mImmediateContext->PSSetShaderResources(85, 1, mOceanDFA->GetAddressOfSRV());
	mImmediateContext->PSSetShaderResources(86, 1, mHeighmap->GetAddressOfSRV());
	mImmediateContext->PSSetShaderResources(84, 1, mHeighmapRawSRV.GetAddressOf());

	return true;
}