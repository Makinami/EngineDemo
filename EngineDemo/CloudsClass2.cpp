#include "CloudsClass2.h"

#include <random>

#include <imgui.h>

#include "DDSTextureLoader.h"
#include "ScreenGrab.h"
#include "Utilities\CreateShader.h"
#include "noise\noise.h"
#include "RenderStates.h"

#include "ShadersManager.h"

using namespace std;
namespace fs = std::experimental::filesystem;

using namespace DirectX;
using namespace noise;
using namespace Microsoft::WRL;

CloudsClass2::CloudsClass2()
	: mScreenQuadIB(0),
	mScreenQuadVB(0),
	mInputLayout(0),
	mVertexShader(nullptr),
	mPixelShader(0),
	cbPerFrameVS(0),
	cbPerFramePS(0),
	mSamplerStateTrilinear(0),
	mDepthStencilState(0)
{
}


CloudsClass2::~CloudsClass2()
{
	ReleaseCOM(mScreenQuadIB);
	ReleaseCOM(mScreenQuadVB);

	ReleaseCOM(mInputLayout);
	ReleaseCOM(mVertexShader);
	ReleaseCOM(mPixelShader);

	ReleaseCOM(cbPerFrameVS);
	ReleaseCOM(cbPerFramePS);

	ReleaseCOM(mSamplerStateTrilinear);
	ReleaseCOM(mDepthStencilState);
}

int CloudsClass2::Init(ID3D11Device1 * device, ID3D11DeviceContext1 * mImmediateContext)
{
	D3D11_TEXTURE3D_DESC text3Desc;

	// general
	text3Desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	text3Desc.CPUAccessFlags = 0;
	text3Desc.Format = DXGI_FORMAT_R16_FLOAT;
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

	ComPtr<ID3D11Texture3D> mCloud;
	device->CreateTexture3D(&text3Desc, 0, &mCloud);
	device->CreateShaderResourceView(mCloud.Get(), &srvDesc, &mCloudGeneralSRV);
	device->CreateUnorderedAccessView(mCloud.Get(), &uavDesc, &mCloudGeneralUAV);

	// detailed
	text3Desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	text3Desc.Width = 32;
	text3Desc.Height = 32;
	text3Desc.Depth = 32;

	srvDesc.Format = text3Desc.Format;

	uavDesc.Format = text3Desc.Format;
	uavDesc.Texture3D.WSize = text3Desc.Depth;

	device->CreateTexture3D(&text3Desc, 0, &mCloud);
	device->CreateShaderResourceView(mCloud.Get(), &srvDesc, &mCloudDetailSRV);
	device->CreateUnorderedAccessView(mCloud.Get(), &uavDesc, &mCloudDetailUAV);


	CreateCSFromFile(L"..\\Debug\\Shaders\\Clouds\\generateGen.cso", device, mGenerateGenCS);
	CreateCSFromFile(L"..\\Debug\\Shaders\\Clouds\\generateDet.cso", device, mGenerateDetCS);

	GenerateSeedGrad(device, mImmediateContext);
	GenerateClouds(mImmediateContext);

	ID3D11Texture2D* srcTex;

	CreateDDSTextureFromFile(device, L"Textures\\inscatter.dds", (ID3D11Resource**)&srcTex, &mCloudCurlSRV, 0, nullptr);
	ReleaseCOM(srcTex);

	CreateDDSTextureFromFile(device, L"Textures\\cloudsTypes.dds", (ID3D11Resource**)&srcTex, &mCloudTypesSRV, 0, nullptr);
	ReleaseCOM(srcTex);

	CreateDDSTextureFromFile(device, L"Textures\\cloudsWeatherFlat.dds", (ID3D11Resource**)&srcTex, &mWeatherSRV, 0, nullptr);
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

	// sampler state
	D3D11_SAMPLER_DESC samplerDesc = {};
	samplerDesc.Filter = D3D11_FILTER_MIN_LINEAR_MAG_MIP_POINT;
	samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.MinLOD = -FLT_MAX;
	samplerDesc.MaxLOD = FLT_MAX;
	samplerDesc.MipLODBias = 0.0f;
	samplerDesc.MaxAnisotropy = 1;
	samplerDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
	samplerDesc.BorderColor[0] =
		samplerDesc.BorderColor[1] =
		samplerDesc.BorderColor[2] =
		samplerDesc.BorderColor[3] = 0.0f;

	device->CreateSamplerState(&samplerDesc, &mSamplerStateTrilinear);

	samplerDesc.Filter = D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT;
	samplerDesc.AddressU =
		samplerDesc.AddressV =
		samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;

	device->CreateSamplerState(&samplerDesc, &mSamplerStateBilinearClamp);

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
	dsDesc.DepthFunc = D3D11_COMPARISON_GREATER_EQUAL;
	dsDesc.StencilEnable = false;

	device->CreateDepthStencilState(&dsDesc, &mDepthStencilState);

	dev = device;

	return 0;
}

void CloudsClass2::Draw(ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight & light, ID3D11ShaderResourceView* transmittanceSRV)
{
	static ShaderMap shaderfiles;
	static int currShader{ 0 };
	static std::string shader{ "/* Shader editor */" };
	shader.reserve(32 * 1024); // for now 32k. I am still looking into arbitrary length textbox (https://github.com/ocornut/imgui/issues/1008)

	if (ImGui::Button("Reload shader list", true))
		shaderfiles = FindShaderFiles("../Debug/Shaders/Clouds/");

	ImGui::SameLine();

	auto reload = (shaderfiles.find(ShaderTypes::Pixel) != std::end(shaderfiles)
		&& shaderfiles[ShaderTypes::Pixel].size() > currShader);
	
	if (ImGui::Button("Reload shader", reload))
	{
		CreatePSFromFile(shaderfiles[ShaderTypes::Pixel][currShader].file, dev, mPixelShader);
	}

	ImGui::SameLine();

	auto editable = (reload
		&& shaderfiles[ShaderTypes::Pixel][currShader].file.extension() == ".hlsl");
	
	if (ImGui::Button("Edit", editable))
	{
		ShellExecute(nullptr, nullptr,
			shaderfiles[ShaderTypes::Pixel][currShader].file.wstring().c_str(), nullptr, nullptr, SW_SHOW);
	}
	if (ImGui::IsItemHovered())
		ImGui::SetTooltip("Edit current shader in external program");

	if (ImGui::Combo("Pixel Shader", &currShader, shaderfiles[ShaderTypes::Pixel]))
	{
		ReleaseCOM(mPixelShader);
		CreatePSFromFile(shaderfiles[ShaderTypes::Pixel][currShader].file, dev, mPixelShader);
	}
	
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

	dataVS->gViewInverse = XMMatrixInverse(nullptr, XMMatrixTranspose(Camera->GetViewRelSun()));
	dataVS->gProjInverse = XMMatrixInverse(nullptr, Camera->GetProjTrans());

	mImmediateContext->Unmap(cbPerFrameVS, 0);

	mImmediateContext->VSSetConstantBuffers(0, 1, &cbPerFrameVS);

	mImmediateContext->VSSetShader(mVertexShader, NULL, 0);

	// PS
	cbPerFramePSType* dataPS;
	mImmediateContext->Map(cbPerFramePS, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	dataPS = (cbPerFramePSType*)mappedResource.pData;

	XMStoreFloat3(&(dataPS->gCameraPos), Camera->GetPosition());
	dataPS->gSunDir = light.Direction();

	mImmediateContext->Unmap(cbPerFramePS, 0);

	mImmediateContext->PSSetConstantBuffers(0, 1, &cbPerFramePS);
	mImmediateContext->PSSetSamplers(3, 1, &mSamplerStateTrilinear);
	mImmediateContext->PSSetSamplers(4, 1, &mSamplerStateBilinearClamp);
	mImmediateContext->PSSetShaderResources(1, 1, &transmittanceSRV);
	mImmediateContext->PSSetShaderResources(4, 1, mCloudGeneralSRV.GetAddressOf());
	mImmediateContext->PSSetShaderResources(5, 1, mCloudDetailSRV.GetAddressOf());
	//mImmediateContext->PSSetShaderResources(6, 1, &mCloudCurlSRV);
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
	ID3D11UnorderedAccessView* ppUAViewNULL[2] = { NULL, NULL };
	mImmediateContext->CSSetShader(mGenerateGenCS, nullptr, 0);
	mImmediateContext->CSSetShaderResources(0, 1, mRandomGradSRV.GetAddressOf());
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, mCloudGeneralUAV.GetAddressOf(), nullptr);
	mImmediateContext->CSSetUnorderedAccessViews(1, 1, mCloudDetailUAV.GetAddressOf(), nullptr);

	mImmediateContext->Dispatch(GEN_RES / 16, GEN_RES / 16, GEN_RES);

	mImmediateContext->CSSetShader(mGenerateDetCS, nullptr, 0);

	mImmediateContext->Dispatch(DET_RES / 16, DET_RES / 16, DET_RES);

	mImmediateContext->CSSetUnorderedAccessViews(0, 2, ppUAViewNULL, nullptr);

	return S_OK;
}

HRESULT CloudsClass2::GenerateSeedGrad(ID3D11Device1 * device, ID3D11DeviceContext1 * mImmediateContext)
{
	char grad[12][4] = { {  1,  1,  0, 0},
						 { -1,  1,  0, 0 },
						 {  1, -1,  0, 0 },
						 { -1, -1,  0, 0 },
						 {  1,  0,  1, 0 },
						 { -1,  0,  1, 0 },
						 {  1,  0, -1, 0 },
						 { -1,  0, -1, 0 },
						 {  0,  1,  1, 0 },
						 {  0, -1,  1, 0 },
						 {  0,  1, -1, 0 },
						 {  0, -1, -1, 0 } };

	BYTE* seed = new BYTE[128*128*128*4];
	
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(0, 11);

	for (auto i = 0; i < 128; ++i)
	{
		for (auto j = 0; j < 128; ++j)
		{
			for (auto k = 0; k < 128; ++k)
			{
				auto id = dis(gen);
				for (auto l = 0; l < 4; ++l)
					seed[4*(i*128*128 + j*128 + k) + l] = grad[id][l];
			}
		}
	}

	D3D11_TEXTURE3D_DESC seedGrad3Desc;
	seedGrad3Desc.Format = DXGI_FORMAT_R8G8B8A8_SINT;
	seedGrad3Desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	seedGrad3Desc.Depth = 128;
	seedGrad3Desc.CPUAccessFlags = 0;
	seedGrad3Desc.Height = 128;
	seedGrad3Desc.MipLevels = 1;
	seedGrad3Desc.MiscFlags = 0;
	seedGrad3Desc.Usage = D3D11_USAGE_IMMUTABLE;
	seedGrad3Desc.Width = 128;

	D3D11_SUBRESOURCE_DATA initData;
	initData.pSysMem = seed;
	initData.SysMemPitch = sizeof(grad[0]) * 128;
	initData.SysMemSlicePitch = sizeof(grad[0]) * 128 * 128;

	D3D11_SHADER_RESOURCE_VIEW_DESC seedGrad3DSRVDesc;
	seedGrad3DSRVDesc.Format = seedGrad3Desc.Format;
	seedGrad3DSRVDesc.Texture3D.MipLevels = 1;
	seedGrad3DSRVDesc.Texture3D.MostDetailedMip = 0;
	seedGrad3DSRVDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE3D;

	device->CreateTexture3D(&seedGrad3Desc, &initData, &mRandomGrad);
	device->CreateShaderResourceView(mRandomGrad.Get(), &seedGrad3DSRVDesc, &mRandomGradSRV);

	delete[] seed;

	return S_OK;
}
