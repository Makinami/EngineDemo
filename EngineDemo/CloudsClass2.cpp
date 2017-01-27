#include "CloudsClass2.h"

#include <vector>
#include <fstream>
#include <algorithm>
#include <random>

#include "DDSTextureLoader.h"
#include "ScreenGrab.h"
#include "Utilities\CreateShader.h"
#include "noise\noise.h"

#include "RenderStates.h"

using namespace std;
using namespace DirectX;
using namespace noise;

CloudsClass2::CloudsClass2()
	: mCloudGeneralSRV(0),
	mCloudGeneralUAV(0),
	mCloudGeneral(0),
	mCloudDetailSRV(0),
	mCloudDetailUAV(0),
	mCloudDetail(0),
	mGenerateCS(0),
	mScreenQuadIB(0),
	mScreenQuadVB(0),
	mInputLayout(0),
	mVertexShader(0),
	mPixelShader(0),
	cbPerFrameVS(0),
	cbPerFramePS(0),
	mDepthStencilState(0),
	mRandomSeed(0),
	mRandomSeedSRV(0),
	mCloudCurlSRV(0),
	mCloudTypesSRV(0),
	mWeatherSRV(0),
	mSamplerStateTrilinear(0),
	mBlendStateClouds(0)

{
}


CloudsClass2::~CloudsClass2()
{
	ReleaseCOM(mCloudGeneralSRV);
	ReleaseCOM(mCloudGeneralUAV);
	ReleaseCOM(mCloudGeneral);

	ReleaseCOM(mCloudDetailSRV);
	ReleaseCOM(mCloudDetailUAV);
	ReleaseCOM(mCloudDetail);

	ReleaseCOM(mCloudCurlSRV);
	ReleaseCOM(mCloudTypesSRV);
	ReleaseCOM(mWeatherSRV);

	ReleaseCOM(mGenerateCS);

	ReleaseCOM(mScreenQuadIB);
	ReleaseCOM(mScreenQuadVB);

	ReleaseCOM(mInputLayout);
	ReleaseCOM(mVertexShader);
	ReleaseCOM(mPixelShader);

	ReleaseCOM(cbPerFrameVS);
	ReleaseCOM(cbPerFramePS);

	ReleaseCOM(mSamplerStateTrilinear);
	ReleaseCOM(mBlendStateClouds);
	ReleaseCOM(mDepthStencilState);

	ReleaseCOM(mRandomSeed);
	ReleaseCOM(mRandomSeedSRV);
}

int CloudsClass2::Init(ID3D11Device1 * device, ID3D11DeviceContext1 * mImmediateContext)
{
	D3D11_TEXTURE3D_DESC text3Desc;

	// general
	text3Desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	text3Desc.CPUAccessFlags = 0;
	text3Desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	text3Desc.Width = GEN_RES;
	text3Desc.Height = GEN_RES;
	text3Desc.Depth = GEN_RES;
	text3Desc.MipLevels = 1;
	text3Desc.MiscFlags = 0;
	text3Desc.Usage = D3D11_USAGE_DEFAULT;

	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
	srvDesc.Format = text3Desc.Format;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE3D;
	srvDesc.Texture3D.MipLevels = 1;
	srvDesc.Texture3D.MostDetailedMip = 0;

	D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc;
	uavDesc.Format = text3Desc.Format;
	uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE3D;
	uavDesc.Texture3D.MipSlice = 0;
	uavDesc.Texture3D.FirstWSlice = 0;
	uavDesc.Texture3D.WSize = text3Desc.Depth;

	device->CreateTexture3D(&text3Desc, 0, &mCloudGeneral);
	device->CreateShaderResourceView(mCloudGeneral, &srvDesc, &mCloudGeneralSRV);
	device->CreateUnorderedAccessView(mCloudGeneral, &uavDesc, &mCloudGeneralUAV);

	// detailed
	text3Desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	text3Desc.Width = 32;
	text3Desc.Height = 32;
	text3Desc.Depth = 32;

	srvDesc.Format = text3Desc.Format;

	uavDesc.Format = text3Desc.Format;
	uavDesc.Texture3D.WSize = text3Desc.Depth;

	device->CreateTexture3D(&text3Desc, 0, &mCloudDetail);
	device->CreateShaderResourceView(mCloudDetail, &srvDesc, &mCloudDetailSRV);
	//device->CreateUnorderedAccessView(mCloudDetail, &uavDesc, &mCloudDetailUAV);

	// random seed
	text3Desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	text3Desc.Width = 128;
	text3Desc.Depth = 128;
	text3Desc.Height = 128;
	text3Desc.Format = DXGI_FORMAT_R32_FLOAT;
	text3Desc.MipLevels = 1;
	text3Desc.MiscFlags = 0;
	text3Desc.Usage = D3D11_USAGE_DEFAULT;

	srvDesc.Format = text3Desc.Format;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE3D;
	srvDesc.Texture3D.MipLevels = 1;
	srvDesc.Texture3D.MostDetailedMip = 0;

	device->CreateTexture3D(&text3Desc, 0, &mRandomSeed);
	device->CreateShaderResourceView(mRandomSeed, &srvDesc, &mRandomSeedSRV);

	CreateCSFromFile(L"..\\Debug\\Shaders\\Clouds\\generate.cso", device, mGenerateCS);

	GenerateClouds(mImmediateContext);

	ID3D11Texture2D* srcTex;

	CreateDDSTextureFromFile(device, L"Textures\\cloudsCurlNoise.dds", (ID3D11Resource**)&srcTex, &mCloudCurlSRV, 0, nullptr);
	ReleaseCOM(srcTex);

	CreateDDSTextureFromFile(device, L"Textures\\cloudsTypes.dds", (ID3D11Resource**)&srcTex, &mCloudTypesSRV, 0, nullptr);
	ReleaseCOM(srcTex);

	CreateDDSTextureFromFile(device, L"Textures\\cloudsWeather.dds", (ID3D11Resource**)&srcTex, &mWeatherSRV, 0, nullptr);
	ReleaseCOM(srcTex);

	// clouds layer quad
	std::vector<DirectX::XMFLOAT3> patchVertices(4);

	patchVertices[0] = XMFLOAT3(-1.0f, -1.0f, 1.0f);
	patchVertices[1] = XMFLOAT3(-1.0f, 1.0f, 1.0f);
	patchVertices[2] = XMFLOAT3(1.0f, 1.0f, 1.0f);
	patchVertices[3] = XMFLOAT3(1.0f, -1.0f, 1.0f);

	D3D11_BUFFER_DESC vbd;
	vbd.Usage = D3D11_USAGE_IMMUTABLE;
	vbd.ByteWidth = sizeof(XMFLOAT3) * patchVertices.size();
	vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vbd.CPUAccessFlags = 0;
	vbd.MiscFlags = 0;
	vbd.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA vinitData;
	vinitData.pSysMem = &patchVertices[0];
	device->CreateBuffer(&vbd, &vinitData, &mScreenQuadVB);

	vector<USHORT> indices(6);

	indices[0] = 0;
	indices[1] = 1;
	indices[2] = 2;
	indices[3] = 0;
	indices[4] = 2;
	indices[5] = 3;

	vbd.ByteWidth = sizeof(USHORT) * indices.size();
	vbd.BindFlags = D3D11_BIND_INDEX_BUFFER;

	vinitData.pSysMem = &indices[0];
	device->CreateBuffer(&vbd, &vinitData, &mScreenQuadIB);

	CreatePSFromFile(L"..\\Debug\\Shaders\\Clouds\\CloudsPS.cso", device, mPixelShader);

	D3D11_INPUT_ELEMENT_DESC vertexDesc[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 }
	};

	int numElements = sizeof(vertexDesc) / sizeof(vertexDesc[0]);

	CreateVSAndInputLayout(L"..\\Debug\\Shaders\\Clouds\\CloudsVS.cso", device, mVertexShader, vertexDesc, numElements, mInputLayout);

	D3D11_BUFFER_DESC cbDesc = {};
	cbDesc.Usage = D3D11_USAGE_DYNAMIC;
	cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	cbDesc.MiscFlags = 0;
	cbDesc.StructureByteStride = 0;
	cbDesc.ByteWidth = sizeof(cbPerFrameVSType);

	device->CreateBuffer(&cbDesc, NULL, &cbPerFrameVS);

	cbDesc.ByteWidth = sizeof(cbPerFramePSType);

	device->CreateBuffer(&cbDesc, NULL, &cbPerFramePS);
	
	// blend state
	D3D11_BLEND_DESC1 blendDesc = {};
	blendDesc.AlphaToCoverageEnable = false;
	blendDesc.IndependentBlendEnable = false;
	blendDesc.RenderTarget[0].BlendEnable = true;
	blendDesc.RenderTarget[0].LogicOpEnable = false;
	blendDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
	blendDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
	blendDesc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
	blendDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
	blendDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
	blendDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
	blendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

	device->CreateBlendState1(&blendDesc, &mBlendStateClouds);

	// depth stencil state
	D3D11_DEPTH_STENCIL_DESC dsDesc;
	dsDesc.DepthEnable = true;
	dsDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
	dsDesc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
	dsDesc.StencilEnable = false;

	device->CreateDepthStencilState(&dsDesc, &mDepthStencilState);

	return 0;
}

void CloudsClass2::Draw(ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight & light, ID3D11ShaderResourceView* transmittanceSRV)
{
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	UINT stride = sizeof(XMFLOAT3);
	UINT offset = 0;

	// IA
	mImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	mImmediateContext->IASetVertexBuffers(0, 1, &mScreenQuadVB, &stride, &offset);
	mImmediateContext->IASetIndexBuffer(mScreenQuadIB, DXGI_FORMAT_R16_UINT, 0);
	mImmediateContext->IASetInputLayout(mInputLayout);

	// VS
	cbPerFrameVSType* dataVS;
	mImmediateContext->Map(cbPerFrameVS, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	dataVS = (cbPerFrameVSType*)mappedResource.pData;

	dataVS->gViewInverse = XMMatrixInverse(nullptr, XMMatrixTranspose(Camera->GetViewMatrix()));
	dataVS->gProjInverse = XMMatrixInverse(nullptr, Camera->GetProjTrans());

	mImmediateContext->Unmap(cbPerFrameVS, 0);

	mImmediateContext->VSSetConstantBuffers(0, 1, &cbPerFrameVS);

	mImmediateContext->VSSetShader(mVertexShader, NULL, 0);

	// PS
	cbPerFramePSType* dataPS;
	mImmediateContext->Map(cbPerFramePS, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	dataPS = (cbPerFramePSType*)mappedResource.pData;

	XMStoreFloat3(&(dataPS->gCameraPos), Camera->GetPositionRelSun());
	dataPS->gSunDir = light.Direction();

	mImmediateContext->Unmap(cbPerFramePS, 0);

	mImmediateContext->PSSetConstantBuffers(0, 1, &cbPerFramePS);
	mImmediateContext->PSSetSamplers(3, 1, &RenderStates::Sampler::TrilinearWrapSS);
	mImmediateContext->PSSetShaderResources(1, 1, &transmittanceSRV);
	mImmediateContext->PSSetShaderResources(4, 1, &mCloudGeneralSRV);
	mImmediateContext->PSSetShaderResources(5, 1, &mCloudDetailSRV);
	mImmediateContext->PSSetShaderResources(6, 1, &mCloudCurlSRV);
	mImmediateContext->PSSetShaderResources(7, 1, &mCloudTypesSRV);
	mImmediateContext->PSSetShaderResources(8, 1, &mWeatherSRV);

	mImmediateContext->PSSetShader(mPixelShader, NULL, 0);

	float blendFactors[] = { 0.0f, 0.0f, 0.0f, 0.0f };
	mImmediateContext->OMSetBlendState(mBlendStateClouds, blendFactors, 0xffffffff);
	mImmediateContext->OMSetDepthStencilState(mDepthStencilState, 0);

	mImmediateContext->DrawIndexed(6, 0, 0);

	mImmediateContext->OMSetBlendState(NULL, NULL, 0xffffff);
	mImmediateContext->OMSetDepthStencilState(0, 0);
}

int CloudsClass2::GenerateClouds(ID3D11DeviceContext1 * mImmediateContext)
{
	bool gpu = false;

	// noise texture generation
	module::Perlin perlinNoise;
	module::Voronoi voronoiNoise;
	module::Billow billowNoise;
	module::Invert invert;
	module::Add add;

	invert.SetSourceModule(0, voronoiNoise);
	add.SetSourceModule(0, billowNoise);
	add.SetSourceModule(1, perlinNoise);

	voronoiNoise.EnableDistance();
	perlinNoise.SetOctaveCount(1);

	std::vector<XMFLOAT4> texture;
	texture.resize(GEN_RES * GEN_RES * GEN_RES);

	std::ifstream ifs;
	std::ofstream ofs;
	uint size = 0;
	uint length = texture.size()*sizeof(XMFLOAT4);
	std::vector<unsigned char> in(length);

	if (gpu)
	{
		// random seed generation
		std::vector<float> seed(128 * 128 * 128);

		std::default_random_engine generator;
		std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

		for (auto i = 0; i < seed.size(); ++i)
			seed[i] = distribution(generator);

		mImmediateContext->UpdateSubresource(mRandomSeed, 0, NULL, &(seed[0]), sizeof(float) * 128, sizeof(float) * 128 * 128);

		ID3D11UnorderedAccessView* ppUAViewNULL[2] = { NULL, NULL };
		ID3D11ShaderResourceView* ppSRVNULL[4] = { NULL, NULL, NULL, NULL };

		mImmediateContext->CSSetUnorderedAccessViews(0, 1, &mCloudGeneralUAV, NULL);
		mImmediateContext->CSSetShaderResources(0, 1, &mRandomSeedSRV);
		mImmediateContext->CSSetShader(mGenerateCS, NULL, 0);

		mImmediateContext->Dispatch(128 / 16, 128, 128 / 16);

		mImmediateContext->CSSetShaderResources(0, 1, ppSRVNULL);
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, NULL);
	}
	else
	{
		ifs.open("cloudsGeneralNoise.raw", std::ifstream::binary);

		if (ifs.good())
		{
			ifs.seekg(0, std::ios::end);
			size = size_t(ifs.tellg());
			ifs.seekg(0, std::ios::beg);
		}

		if (size >= length)
		{
			ifs.read((char*)&(texture[0]), texture.size()*sizeof(XMFLOAT4));
		}
		else
		{
			ofs.open("cloudsGeneralNoise.raw", std::ofstream::trunc);
			ofs.close();
			ofs.open("cloudsGeneralNoise.raw", std::ofstream::binary);

			ofs.good();

			ofs.is_open();

			for (int i = 0; i < GEN_RES; ++i)
			{
				for (int j = 0; j < GEN_RES; ++j)
				{
					for (int k = 0; k < GEN_RES; ++k)
					{
						int index = i*GEN_RES *GEN_RES + j*GEN_RES + k;
						voronoiNoise.SetFrequency(module::DEFAULT_VORONOI_FREQUENCY);
						texture[index].x = max(0.0f, min(1.0f, perlinNoise.GetValue(i * 5 / 128.0f, j * 5 / 128.0f, k * 5 / 128.0f)));
						texture[index].y = 0;// max(0, voronoiNoise.GetValue(i * 33 / 128.0f + 13 / 3.0f, j * 33 / 128.0f + 17 / 5.0f, k * 33 / 128.0f + 19 / 7.0f));
						voronoiNoise.SetFrequency(voronoiNoise.GetFrequency() * 2);
						texture[index].z = 0;// (1.0f + voronoiNoise.GetValue(i * 33 / 128.0f + 19 / 3.0f, j * 33 / 128.0f + 23 / 5.0f, k * 33 / 128.0f + 27 / 7.0f)) / 2.0f;
						voronoiNoise.SetFrequency(voronoiNoise.GetFrequency() * 2);
						texture[index].w = 0;// (1.0f + voronoiNoise.GetValue(i * 33 / 128.0f + 31 / 3.0f, j * 33 / 128.0f + 37 / 5.0f, k * 33 / 128.0f + 41 / 7.0f)) / 2.0f;
					}
				}
			}

			ofs.write((const char*)&(texture[0]), length);
			ofs.flush();
		}

		ifs.close();
		ofs.close();

		mImmediateContext->UpdateSubresource(mCloudGeneral, 0, NULL, &(texture[0]), sizeof(XMFLOAT4) * 128, sizeof(XMFLOAT4) * 128 * 128);
	}
	
	texture.resize(DET_RES * DET_RES * DET_RES);
	length = texture.size()* sizeof(XMFLOAT4);
	size = 0;

	ifs.open("cloudsDetailNoise.raw", std::ifstream::binary);

	if (ifs.good())
	{
		ifs.seekg(0, std::ios::end);
		size = size_t(ifs.tellg());
		ifs.seekg(0, std::ios::beg);
	}

	if (size >= length)
	{
		ifs.read((char*)&(texture[0]), length);
	}
	else
	{
		ofs.open("cloudsDetailNoise.raw", std::ofstream::trunc);
		ofs.close();
		ofs.open("cloudsDetailNoise.raw", std::ofstream::binary);

		ofs.good();

		ofs.is_open();

		for (int i = 0; i < DET_RES; ++i)
		{
			for (int j = 0; j < DET_RES; ++j)
			{
				for (int k = 0; k < DET_RES; ++k)
				{
					int index = i*DET_RES*DET_RES + j*DET_RES + k;
					voronoiNoise.SetFrequency(module::DEFAULT_VORONOI_FREQUENCY);
					texture[index].x = billowNoise.GetValue(i * 5 / 32.0f, j * 5 / 32.0f, k * 5 / 32.0f);
					voronoiNoise.SetFrequency(voronoiNoise.GetFrequency() * 2);
					texture[index].y = 0; // (1.0f + voronoiNoise.GetValue(i * 33 / 32.0f + 13 / 3.0f, j * 33 / 32.0f + 17 / 5.0f, k * 33 / 32.0f + 19 / 7.0f)) / 2.0f;
					voronoiNoise.SetFrequency(voronoiNoise.GetFrequency() * 2);
					texture[index].y = 0; // (1.0f + voronoiNoise.GetValue(i * 33 / 32.0f + 19 / 3.0f, j * 33 / 32.0f + 21 / 5.0f, k * 33 / 32.0f + 27 / 7.0f)) / 2.0f;
					texture[index].z = 0;
				}
			}
		}

		ofs.write((const char*)&(texture[0]), length);
		ofs.flush();
	}

	ifs.close();
	ofs.close();	

	mImmediateContext->UpdateSubresource(mCloudDetail, 0, NULL, &(texture[0]), sizeof(XMFLOAT4)*DET_RES, sizeof(XMFLOAT4)*DET_RES*DET_RES);

	return 0;
}
