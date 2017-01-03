#include "CloudsClass.h"

#include <vector>

#include "Utilities\CreateShader.h"
#include "RenderStates.h"

using namespace std;
using namespace DirectX;

CloudsClass::CloudsClass() :
	mCloudSRV(0),
	mCloudUAV(0),
	mGenerateCS(0),
	mCloudsQuadIB(0),
	mCloudsQuadVB(0),
	mInputLayout(0),
	mVertexShader(0),
	mPixelShader(0),
	cbPerFrameVS(0),
	cbPerFramePS(0)
{
}


CloudsClass::~CloudsClass()
{
	ReleaseCOM(mCloudSRV);
	ReleaseCOM(mCloudUAV);

	ReleaseCOM(mGenerateCS);

	ReleaseCOM(mCloudsQuadIB);
	ReleaseCOM(mCloudsQuadVB);

	ReleaseCOM(mInputLayout);
	ReleaseCOM(mVertexShader);
	ReleaseCOM(mPixelShader);

	ReleaseCOM(cbPerFrameVS);
	ReleaseCOM(cbPerFramePS);
}

int CloudsClass::Init(ID3D11Device1 * device, ID3D11DeviceContext1 * mImmediateContext)
{
	D3D11_TEXTURE3D_DESC text3Desc;
	text3Desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	text3Desc.CPUAccessFlags = 0;
	text3Desc.Depth = 128;
	text3Desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	text3Desc.Height = 10;
	text3Desc.MipLevels = 1;
	text3Desc.MiscFlags = 0;
	text3Desc.Usage = D3D11_USAGE_DEFAULT;
	text3Desc.Width = 128;

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

	ID3D11Texture3D* text3D;
	device->CreateTexture3D(&text3Desc, 0, &text3D);
	device->CreateShaderResourceView(text3D, &srvDesc, &mCloudSRV);
	device->CreateUnorderedAccessView(text3D, &uavDesc, &mCloudUAV);

	ReleaseCOM(text3D);

	CreateCSFromFile(L"..\\Debug\\Shaders\\Clouds\\generate.cso", device, mGenerateCS);

	GenerateClouds(mImmediateContext);

	// clouds layer quad
	std::vector<DirectX::XMFLOAT3> patchVertices(4);

	patchVertices[0] = XMFLOAT3(-1.0f, 0.0f, -1.0f);
	patchVertices[1] = XMFLOAT3(-1.0f, 0.0f, 1.0f);
	patchVertices[2] = XMFLOAT3(1.0f, 0.0f, 1.0f);
	patchVertices[3] = XMFLOAT3(1.0f, 0.0f, -1.0f);

	D3D11_BUFFER_DESC vbd;
	vbd.Usage = D3D11_USAGE_IMMUTABLE;
	vbd.ByteWidth = sizeof(XMFLOAT3) * patchVertices.size();
	vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vbd.CPUAccessFlags = 0;
	vbd.MiscFlags = 0;
	vbd.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA vinitData;
	vinitData.pSysMem = &patchVertices[0];
	device->CreateBuffer(&vbd, &vinitData, &mCloudsQuadVB);

	vector<USHORT> indices(6);

	indices[0] = 0;
	indices[1] = 3;
	indices[2] = 2;
	indices[3] = 0;
	indices[4] = 2;
	indices[5] = 1;

	vbd.ByteWidth = sizeof(USHORT) * indices.size();
	vbd.BindFlags = D3D11_BIND_INDEX_BUFFER;

	vinitData.pSysMem = &indices[0];
	device->CreateBuffer(&vbd, &vinitData, &mCloudsQuadIB);

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

	return 0;
}

void CloudsClass::Draw(ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight & light)
{
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	UINT stride = sizeof(XMFLOAT3);
	UINT offset = 0;

	// IA
	mImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	mImmediateContext->IASetVertexBuffers(0, 1, &mCloudsQuadVB, &stride, &offset);
	mImmediateContext->IASetIndexBuffer(mCloudsQuadIB, DXGI_FORMAT_R16_UINT, 0);
	mImmediateContext->IASetInputLayout(mInputLayout);

	// VS
	XMMATRIX mWorld = XMMatrixScaling(1024.0f, 1.0f, 1024.0f)*XMMatrixTranslation(0.0f, 500.0f, 0.0f);


	cbPerFrameVSType* dataVS;
	mImmediateContext->Map(cbPerFrameVS, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	dataVS = (cbPerFrameVSType*)mappedResource.pData;

	dataVS->gProjView = Camera->GetViewProjTransMatrix();
	dataVS->gWorld = XMMatrixTranspose(mWorld);

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
	mImmediateContext->PSSetShaderResources(0, 1, &mCloudSRV);
	mImmediateContext->PSSetSamplers(3, 1, &RenderStates::Sampler::TrilinearClampSS);

	mImmediateContext->PSSetShader(mPixelShader, NULL, 0);

	float blendFactors[] = { 0.0f, 0.0f, 0.0f, 0.0f };
	mImmediateContext->OMSetBlendState(mBlendStateClouds, blendFactors, 0xffffffff);

	mImmediateContext->DrawIndexed(6, 0, 0);

	mImmediateContext->OMSetBlendState(NULL, NULL, 0xffffff);
}

int CloudsClass::GenerateClouds(ID3D11DeviceContext1 * mImmediateContext)
{
	ID3D11UnorderedAccessView* ppUAVNULL[] = { NULL };
	ID3D11ShaderResourceView* ppSRVNULL[] = { NULL };

	mImmediateContext->CSSetUnorderedAccessViews(0, 1, &mCloudUAV, NULL);
	mImmediateContext->CSSetShader(mGenerateCS, NULL, 0);

	mImmediateContext->Dispatch(128 / 16, 1, 128 / 16);

	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAVNULL, NULL);

	return 0;
}
