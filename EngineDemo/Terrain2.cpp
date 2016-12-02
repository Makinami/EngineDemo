#include "Terrain2.h"

#include "ShaderManager.h"
#include "DDSTextureLoader.h"
#include "ScreenGrab.h"
#include "Utilities\CreateBuffer.h"
#include "Utilities\MapResources.h"
#include "Utilities\CreateShader.h"
#include "RenderStates.h"

using namespace DirectX;

TerrainClass2::TerrainClass2()
{
}


TerrainClass2::~TerrainClass2()
{
}

bool TerrainClass2::Init(ID3D11Device1 * device, ID3D11DeviceContext1 * mImmediateContext)
{
	mInitJFA = ShaderManager::Instance()->GetCS("Terrain2::initJFA");
	mStepJFA = ShaderManager::Instance()->GetCS("Terrain2::stepJFA");
	mPostJFA = ShaderManager::Instance()->GetCS("Terrain2::postJFA");

	mProcessHM = ShaderManager::Instance()->GetCS("Terrain2::procesHeighmap");

	CreateDDSTextureFromFile(device, L"lake.dds", nullptr, &mHeighmapRawSRV);
	CreateDDSTextureFromFile(device, L"processedHeighmap.dds", nullptr, &mProDF);
	//mImmediateContext->VSSetShaderResources(40, 1, mProDF.GetAddressOf());

	D3D11_TEXTURE2D_DESC textDesc;

	ZeroMemory(&textDesc, sizeof(textDesc));

	textDesc.Width = 4096;
	textDesc.Height = 4096;
	textDesc.MipLevels = 0;
	textDesc.ArraySize = 1;
	textDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	textDesc.SampleDesc = { 1, 0 };
	textDesc.Usage = D3D11_USAGE_DEFAULT;
	textDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	textDesc.CPUAccessFlags = 0;
	textDesc.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;

	EXIT_ON_NULL(mOceanDFA = 
				 TextureFactory::CreateTexture(textDesc));

	EXIT_ON_NULL(mOceanDFB =
				 TextureFactory::CreateTexture(textDesc));

	EXIT_ON_NULL(mHeighmap =
				 TextureFactory::CreateTexture(D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
											   DXGI_FORMAT_R16_FLOAT,
											   4096, 4096));

	CreateConstantBuffer(device, sizeof(JFAParams), mJFACB);

	ID3D11UnorderedAccessView* ppUAViewNULL[2] = { NULL, NULL };
	ID3D11ShaderResourceView* ppSRVNULL[4] = { NULL, NULL, NULL, NULL };

	JFAParams.size[0] = 4096;
	JFAParams.size[1] = 4096;
	JFAParams.mip = log2(4096 / 4096);
	MapResources(mImmediateContext, mJFACB.Get(), JFAParams);
	mImmediateContext->CSSetConstantBuffers(0, 1, mJFACB.GetAddressOf());

	mImmediateContext->CSSetShaderResources(0, 1, mHeighmapRawSRV.GetAddressOf());
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, mOceanDFA->GetAddressOfUAV(), nullptr);
	mImmediateContext->CSSetShader(mInitJFA, nullptr, 0);

	mImmediateContext->Dispatch(4096 / 16, 4096 / 16, 1);

	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);

	mImmediateContext->CSSetShader(mStepJFA, nullptr, 0);
	JFAParams.mip = 4096.0f / 4096.0f;

	for (int step = 2048; step; step /= 2)
	{
		JFAParams.step = step;
		MapResources(mImmediateContext, mJFACB.Get(), JFAParams);
		mImmediateContext->CSSetConstantBuffers(0, 1, mJFACB.GetAddressOf());

		mImmediateContext->CSSetShaderResources(0, 1, mOceanDFA->GetAddressOfSRV());
		mImmediateContext->CSSetUnorderedAccessViews(1, 1, mOceanDFB->GetAddressOfUAV(), 0);

		mImmediateContext->Dispatch(4096 / 16, 4096 / 16, 1);

		mImmediateContext->CSSetUnorderedAccessViews(0, 2, ppUAViewNULL, 0);

		std::swap(mOceanDFA, mOceanDFB);
	}

	mImmediateContext->CSSetShaderResources(0, 1, mHeighmapRawSRV.GetAddressOf());
	mImmediateContext->CSSetShaderResources(1, 1, mOceanDFA->GetAddressOfSRV());
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, mHeighmap->GetAddressOfUAV(), nullptr);
	mImmediateContext->CSSetShader(mProcessHM, nullptr, 0);

	mImmediateContext->Dispatch(4096 / 16, 4096 / 16, 1);

	mImmediateContext->CSSetUnorderedAccessViews(0, 2, ppUAViewNULL, nullptr);

	JFAParams.mip = 4096.0f / 4096.0f;
	MapResources(mImmediateContext, mJFACB.Get(), JFAParams);
	mImmediateContext->CSSetConstantBuffers(0, 1, mJFACB.GetAddressOf());

	mImmediateContext->CSSetShaderResources(1, 1, mHeighmap->GetAddressOfSRV());
	mImmediateContext->CSSetShaderResources(0, 1, mOceanDFA->GetAddressOfSRV());
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, mOceanDFB->GetAddressOfUAV(), 0);
	mImmediateContext->CSSetShader(mPostJFA, nullptr, 0);

	mImmediateContext->Dispatch(4096 / 16, 4096 / 16, 1);

	mImmediateContext->CSSetUnorderedAccessViews(0, 2, ppUAViewNULL, nullptr);

	std::swap(mOceanDFA, mOceanDFB);

	mImmediateContext->GenerateMips(mOceanDFA->GetSRV());

	mImmediateContext->VSSetShaderResources(40, 1, mOceanDFA->GetAddressOfSRV());
	mImmediateContext->PSSetShaderResources(40, 1, mOceanDFA->GetAddressOfSRV());
	mImmediateContext->PSSetShaderResources(86, 1, mHeighmap->GetAddressOfSRV());
	mImmediateContext->PSSetShaderResources(84, 1, mHeighmapRawSRV.GetAddressOf());

	terrainQuadTree.Init(device, -2048.0f, -2048.0f, 4096.0f, 4096.0f, 7, 20, XMFLOAT3(1.0, 80.0, 1.0));

	CreateConstantBuffer(device, sizeof(MatrixBuffer), mMatrixCB);

	mPixelShader = ShaderManager::Instance()->GetPS("Terrain2::TerrainPS");
	
	// vs & il quad
	const D3D11_INPUT_ELEMENT_DESC vertexQuadDesc[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "PSIZE", 0, DXGI_FORMAT_R32_FLOAT, 1, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
		{ "PSIZE", 1, DXGI_FORMAT_R32_FLOAT, 1, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
		{ "BLENDINDICES", 0, DXGI_FORMAT_R32_UINT, 1, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
		{ "PSIZE", 2, DXGI_FORMAT_R32_FLOAT, 1, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_INSTANCE_DATA, 1 }
	};

	UINT numElements = sizeof(vertexQuadDesc) / sizeof(vertexQuadDesc[0]);

	CreateVSAndInputLayout(L"..\\Debug\\Shaders\\Terrain2\\TerrainVS.cso", device, mVertexShader, vertexQuadDesc, numElements, mQuadIL);

	return true;
}

void TerrainClass2::Draw(ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight & light)
{
	terrainQuadTree.GenerateTree(mImmediateContext, Camera);

	XMMATRIX ViewProjTrans = Camera->GetViewProjTransMatrix();
	XMStoreFloat3(&MatrixBuffer.camPos, Camera->GetPosition());
	MatrixBuffer.gWorldProj = ViewProjTrans;
	MapResources(mImmediateContext, mMatrixCB.Get(), MatrixBuffer);
	
	// IA
	mImmediateContext->IASetInputLayout(mQuadIL.Get());
	mImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	// VS
	mImmediateContext->VSSetShader(mVertexShader.Get(), nullptr, 0);
	mImmediateContext->VSSetConstantBuffers(4, 1, mMatrixCB.GetAddressOf());
	mImmediateContext->VSSetShaderResources(10, 1, mHeighmap->GetAddressOfSRV());

	// PS
	mImmediateContext->PSSetShader(mPixelShader, nullptr, 0);

	// RS
	mImmediateContext->RSSetState(RenderStates::Rasterizer::DefaultRS);

	terrainQuadTree.Draw(mImmediateContext);

	mImmediateContext->RSSetState(RenderStates::Rasterizer::DefaultRS);
}
