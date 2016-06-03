#include "Sky.h"
#include "Shaders\Sky\resolutions.h"

#include "Utilities\CreateShader.h"
#include "Utilities\RenderViewTargetStack.h"
#include "Utilities\CreateBuffer.h"
#include "Utilities\MapResources.h"

#include <DirectXColors.h>

SkyClass::SkyClass()
	: mRastStateBasic(nullptr),
	transmitanceSRV(0),
	transmitanceUAV(0),
	transmittanceText(0),
	deltaESRV(0),
	deltaEUAV(0),
	irradianceSRV(0),
	irradianceUAV(0),
	irradianceText(0),
	mapRTV(0),
	mapSRV(0),
	skyMapSize(512),
	copyIrradianceSRV(nullptr)
{
	deltaSSRV = new ID3D11ShaderResourceView*[2];
	deltaSUAV = new ID3D11UnorderedAccessView*[2];
	
	for (size_t i = 0; i < 2; ++i)
	{
		deltaSSRV[i] = 0;
		deltaSUAV[i] = 0;
	}
}

SkyClass::~SkyClass()
{
	ReleaseCOM(transmitanceSRV);
	ReleaseCOM(transmitanceUAV);
	ReleaseCOM(transmittanceText);

	ReleaseCOM(deltaESRV);
	ReleaseCOM(deltaEUAV);

	for (size_t i = 0; i < 2; i++)
	{
		ReleaseCOM(deltaSSRV[i]);
		ReleaseCOM(deltaSUAV[i]);
	}
	delete[] deltaSSRV;
	delete[] deltaSUAV;

	ReleaseCOM(irradianceSRV);
	ReleaseCOM(irradianceUAV);
	ReleaseCOM(irradianceText);
	ReleaseCOM(copyIrradianceSRV);
	ReleaseCOM(copyIrradianceUAV);
	ReleaseCOM(copyIrradianceText);

	ReleaseCOM(inscatterSRV);
	ReleaseCOM(inscatterUAV);
	ReleaseCOM(inscatterText);
	ReleaseCOM(copyInscatterSRV);
	ReleaseCOM(copyInscatterUAV);
	ReleaseCOM(copyInscatterText);

	ReleaseCOM(deltaJSRV);
	ReleaseCOM(deltaJUAV);

	ReleaseCOM(transmittanceCS);
	ReleaseCOM(irradiance1CS);
	ReleaseCOM(inscatter1CS);
	ReleaseCOM(copyInscatter1CS);
	ReleaseCOM(inscatterSCS);
	ReleaseCOM(irradianceNCS);
	ReleaseCOM(inscatterNCS);
	ReleaseCOM(copyIrradianceCS);
	ReleaseCOM(copyInscatterNCS);

	ReleaseCOM(cbNOrder);
	ReleaseCOM(cbPerFramePS);
	ReleaseCOM(cbPerFrameVS);

	ReleaseCOM(mScreenQuadIB);
	ReleaseCOM(mScreenQuadVB);

	ReleaseCOM(mInputLayout);
	ReleaseCOM(mVertexShader);
	ReleaseCOM(mPixelShaderToCube);
	ReleaseCOM(mPixelShaderToScreen);

	ReleaseCOM(mRastStateBasic);
	// release only first (rest point to the same resource)
	ReleaseCOM(mSamplerStateBasic[0]);
	delete[] mSamplerStateBasic;
	ReleaseCOM(mSamplerStateTrilinear);
	ReleaseCOM(mDepthStencilStateSky);
	
}

int SkyClass::Init(ID3D11Device1 * device, ID3D11DeviceContext1 * mImmediateContext)
{
	D3D11_TEXTURE2D_DESC text2Desc;
	text2Desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	text2Desc.CPUAccessFlags = 0;
	text2Desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	text2Desc.MipLevels = 1;
	text2Desc.MiscFlags = 0;
	text2Desc.SampleDesc.Count = 1;
	text2Desc.SampleDesc.Quality = 0;
	text2Desc.Usage = D3D11_USAGE_DEFAULT;
	
	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
	srvDesc.Format = text2Desc.Format;

	D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc;
	uavDesc.Format = text2Desc.Format;

	// 2D
	ID3D11Texture2D* text2D;

	text2Desc.ArraySize = 1;

	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MostDetailedMip = 0;
	srvDesc.Texture2D.MipLevels = 1;

	uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
	uavDesc.Texture2D.MipSlice = 0;

	// transmitance
	text2Desc.Height = TRANSMITTANCE_H;
	text2Desc.Width = TRANSMITTANCE_W;

	device->CreateTexture2D(&text2Desc, 0, &transmittanceText);
	device->CreateShaderResourceView(transmittanceText, &srvDesc, &transmitanceSRV);
	device->CreateUnorderedAccessView(transmittanceText, &uavDesc, &transmitanceUAV);

	// deltaE
	text2Desc.Height = SKY_H;
	text2Desc.Width = SKY_W;

	device->CreateTexture2D(&text2Desc, 0, &text2D);
	device->CreateShaderResourceView(text2D, &srvDesc, &deltaESRV);
	device->CreateUnorderedAccessView(text2D, &uavDesc, &deltaEUAV);

	// irradiance
	device->CreateTexture2D(&text2Desc, 0, &irradianceText);
	device->CreateShaderResourceView(irradianceText, &srvDesc, &irradianceSRV);
	device->CreateUnorderedAccessView(irradianceText, &uavDesc, &irradianceUAV);

	device->CreateTexture2D(&text2Desc, 0, &copyIrradianceText);
	device->CreateShaderResourceView(copyIrradianceText, &srvDesc, &copyIrradianceSRV);
	device->CreateUnorderedAccessView(copyIrradianceText, &uavDesc, &copyIrradianceUAV);

	// 3D
	D3D11_TEXTURE3D_DESC text3Desc;
	text3Desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	text3Desc.CPUAccessFlags = 0;
	text3Desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	text3Desc.MipLevels = 1;
	text3Desc.MiscFlags = 0;
	text3Desc.Usage = D3D11_USAGE_DEFAULT;

	ID3D11Texture3D* text3D;
	
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE3D;
	srvDesc.Texture3D.MostDetailedMip = 0;
	srvDesc.Texture3D.MipLevels = 1;

	uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE3D;
	uavDesc.Texture3D.MipSlice = 0;
	uavDesc.Texture3D.FirstWSlice = 0;

	// deltaS
	text3Desc.Height = RES_MU;
	text3Desc.Width = RES_MU_S*RES_NU;
	text3Desc.Depth = RES_R;

	uavDesc.Texture3D.WSize = RES_R;
	
	device->CreateTexture3D(&text3Desc, 0, &text3D);
	device->CreateShaderResourceView(text3D, &srvDesc, &deltaSSRV[0]);
	device->CreateUnorderedAccessView(text3D, &uavDesc, &deltaSUAV[0]);
	ReleaseCOM(text3D);

	device->CreateTexture3D(&text3Desc, 0, &text3D);
	device->CreateShaderResourceView(text3D, &srvDesc, &deltaSSRV[1]);
	device->CreateUnorderedAccessView(text3D, &uavDesc, &deltaSUAV[1]);
	ReleaseCOM(text3D);

	// inscatter
	device->CreateTexture3D(&text3Desc, 0, &inscatterText);
	device->CreateShaderResourceView(inscatterText, &srvDesc, &inscatterSRV);
	device->CreateUnorderedAccessView(inscatterText, &uavDesc, &inscatterUAV);

	device->CreateTexture3D(&text3Desc, 0, &copyInscatterText);
	device->CreateShaderResourceView(copyInscatterText, &srvDesc, &copyInscatterSRV);
	device->CreateUnorderedAccessView(copyInscatterText, &uavDesc, &copyInscatterUAV);

	// deltaJ
	device->CreateTexture3D(&text3Desc, 0, &text3D);
	device->CreateShaderResourceView(text3D, &srvDesc, &deltaJSRV);
	device->CreateUnorderedAccessView(text3D, &uavDesc, &deltaJUAV);
	

	CreateCSFromFile(L"..\\Debug\\Shaders\\Sky\\transmittance.cso", device, transmittanceCS);
	CreateCSFromFile(L"..\\Debug\\Shaders\\Sky\\irradiance1.cso", device, irradiance1CS);
	CreateCSFromFile(L"..\\Debug\\Shaders\\Sky\\inscatter1.cso", device, inscatter1CS);
	CreateCSFromFile(L"..\\Debug\\Shaders\\Sky\\copyInscatter1.cso", device, copyInscatter1CS);
	CreateCSFromFile(L"..\\Debug\\Shaders\\Sky\\inscatterS.cso", device, inscatterSCS);
	CreateCSFromFile(L"..\\Debug\\Shaders\\Sky\\irradianceN.cso", device, irradianceNCS);
	CreateCSFromFile(L"..\\Debug\\Shaders\\Sky\\inscatterN.cso", device, inscatterNCS);
	CreateCSFromFile(L"..\\Debug\\Shaders\\Sky\\copyIrradiance.cso", device, copyIrradianceCS);
	CreateCSFromFile(L"..\\Debug\\Shaders\\Sky\\copyInscatterN.cso", device, copyInscatterNCS);

	// sampler states
	D3D11_SAMPLER_DESC samplerDesc = {};
	samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.MinLOD = -FLT_MAX;
	samplerDesc.MaxLOD = FLT_MAX;
	samplerDesc.MipLODBias = 0.0f;
	samplerDesc.MaxAnisotropy = 1;
	samplerDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
	samplerDesc.BorderColor[0] =
		samplerDesc.BorderColor[1] = 
		samplerDesc.BorderColor[2] =
		samplerDesc.BorderColor[3] = 0.0f;
	
	mSamplerStateBasic = new ID3D11SamplerState*[4];
	device->CreateSamplerState(&samplerDesc, &mSamplerStateBasic[0]);
	for (size_t i = 1; i < 4; i++) mSamplerStateBasic[i] = mSamplerStateBasic[0];

	samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	device->CreateSamplerState(&samplerDesc, &mSamplerStateTrilinear);

	samplerDesc.Filter = D3D11_FILTER_ANISOTROPIC;
	samplerDesc.MaxAnisotropy = 16;
	device->CreateSamplerState(&samplerDesc, &mSamplerAnisotropic);

	// depth stencil state
	D3D11_DEPTH_STENCIL_DESC dsDesc;
	dsDesc.DepthEnable = true;
	dsDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
	dsDesc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
	dsDesc.StencilEnable = false;
	
	device->CreateDepthStencilState(&dsDesc, &mDepthStencilStateSky);

	D3D11_BUFFER_DESC cbDesc = {};
	cbDesc.Usage = D3D11_USAGE_DYNAMIC;
	cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	cbDesc.MiscFlags = 0;
	cbDesc.StructureByteStride = 0;

	// rastarizer state
	D3D11_RASTERIZER_DESC rastDesc;
	ZeroMemory(&rastDesc, sizeof(D3D11_RASTERIZER_DESC));
	rastDesc.FillMode = D3D11_FILL_SOLID;
	rastDesc.CullMode = D3D11_CULL_BACK;
	rastDesc.FrontCounterClockwise = false;
	rastDesc.DepthClipEnable = true;

	if (FAILED(device->CreateRasterizerState(&rastDesc, &mRastStateBasic))) return false;
	
	// precompute buffer
	cbDesc.ByteWidth = sizeof(cbNOrderType);

	device->CreateBuffer(&cbDesc, NULL, &cbNOrder);

	Precompute(mImmediateContext);

	ReleaseCOM(text2D);
	ReleaseCOM(text3D);

	// render's buffers
	cbDesc.ByteWidth = sizeof(cbPerFrameVSType);
	
	device->CreateBuffer(&cbDesc, NULL, &cbPerFrameVS);

	cbDesc.ByteWidth = sizeof(cbPerFramePSType);

	device->CreateBuffer(&cbDesc, NULL, &cbPerFramePS);

	// full screen quad
	vector<XMFLOAT3> patchVertices(4);
	
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

	vbd.ByteWidth = sizeof(USHORT)*indices.size();
	vbd.BindFlags = D3D11_BIND_INDEX_BUFFER;

	vinitData.pSysMem = &indices[0];
	device->CreateBuffer(&vbd, &vinitData, &mScreenQuadIB);

	CreatePSFromFile(L"..\\Debug\\Shaders\\Sky\\SkyToCubePS.cso", device, mPixelShaderToCube);

	CreatePSFromFile(L"..\\Debug\\Shaders\\Sky\\SkyToScreenPS.cso", device, mPixelShaderToScreen, "SkyToScreenPS - mPixelShaderToScreen");

	D3D11_INPUT_ELEMENT_DESC vertexDesc[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 }
	};

	int numElements = sizeof(vertexDesc) / sizeof(vertexDesc[0]);

	CreateVSAndInputLayout(L"..\\Debug\\Shaders\\Sky\\SkyVS.cso", device, mVertexShader, vertexDesc, numElements, mInputLayout);


	/* Cube Map */
	ID3D11Texture2D* cubeText;
	
	// texture
	text2Desc.Width = CubeMapSize;
	text2Desc.Height = CubeMapSize;
	text2Desc.MipLevels = 0;
	text2Desc.ArraySize = 6;
	text2Desc.SampleDesc.Count = 1;
	text2Desc.SampleDesc.Quality = 0;
	text2Desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	text2Desc.Usage = D3D11_USAGE_DEFAULT;
	text2Desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
	text2Desc.CPUAccessFlags = 0;
	text2Desc.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS | D3D11_RESOURCE_MISC_TEXTURECUBE;

	device->CreateTexture2D(&text2Desc, 0, &cubeText);

	// render target view
	cubeTextRTV = new ID3D11RenderTargetView*[6];

	D3D11_RENDER_TARGET_VIEW_DESC rtvDesc;
	rtvDesc.Format = text2Desc.Format;
	rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
	rtvDesc.Texture2DArray.MipSlice = 0;

	// only create a view to one array element
	rtvDesc.Texture2DArray.ArraySize = 1;

	for (int i = 0; i < 6; ++i)
	{
		// create a render target view to the ith element
		rtvDesc.Texture2DArray.FirstArraySlice = i;
		device->CreateRenderTargetView(cubeText, &rtvDesc, &cubeTextRTV[i]);
	}

	// shader resources view
	srvDesc.Format = text2Desc.Format;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBE;
	srvDesc.TextureCube.MostDetailedMip = 0;
	srvDesc.TextureCube.MipLevels = -1;

	device->CreateShaderResourceView(cubeText, &srvDesc, &cubeTextSRV);

	ReleaseCOM(cubeText);

	// depth texture
	text2Desc.MipLevels = 1;
	text2Desc.ArraySize = 1;
	text2Desc.Format = DXGI_FORMAT_D32_FLOAT;
	text2Desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
	text2Desc.MiscFlags = 0;

	ID3D11Texture2D* depthText = 0;
	device->CreateTexture2D(&text2Desc, 0, &depthText);

	// depth view
	D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc;
	dsvDesc.Format = text2Desc.Format;
	dsvDesc.Flags = 0;
	dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	dsvDesc.Texture2D.MipSlice = 0;
	device->CreateDepthStencilView(depthText, &dsvDesc, &cubeMapDSV);

	ReleaseCOM(depthText);

	cubeViewport.TopLeftX = 0.0f;
	cubeViewport.TopLeftY = 0.0f;
	cubeViewport.Width = (float)CubeMapSize;
	cubeViewport.Height = (float)CubeMapSize;
	cubeViewport.MinDepth = 0.0f;
	cubeViewport.MaxDepth = 1.0f;
	
	BuildCubeFaceCamera(0.0f, 6360.0f, 0.0f);

	/*
	 * Sky Map
	 */
	ID3D11Texture2D* mapText;
	// sky map
	text2Desc.ArraySize = 1;
	text2Desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
	text2Desc.CPUAccessFlags = 0;
	text2Desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	text2Desc.Height = skyMapSize;
	text2Desc.MipLevels = 0;
	text2Desc.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;
	text2Desc.SampleDesc = { 1, 0 };
	text2Desc.Usage = D3D11_USAGE_DEFAULT;
	text2Desc.Width = skyMapSize;

	device->CreateTexture2D(&text2Desc, nullptr, &mapText);
	// rendet target view
	rtvDesc.Format = text2Desc.Format;
	rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
	rtvDesc.Texture2D.MipSlice = 0;

	device->CreateRenderTargetView(mapText, &rtvDesc, &mapRTV);

	// srv
	srvDesc.Format = text2Desc.Format;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MipLevels = -1;
	srvDesc.Texture2D.MostDetailedMip = 0;

	device->CreateShaderResourceView(mapText, &srvDesc, &mapSRV);

	ReleaseCOM(mapText);

	CreateVSFromFile(L"..\\Debug\\Shaders\\Sky\\SkyMapVS.cso", device, mMapVertexShader, "SkyMapVS - mMapVertexShader");

	CreatePSFromFile(L"..\\Debug\\Shaders\\Sky\\SkyMapPS.cso", device, mMapPixelShader, "SkyMapPS - mMapPixelShader");

	CreateConstantBuffer(device, sizeof(skyMapBufferType), skyMapCB, "skyMapCB");

	drawSky = Performance->ReserveName(L"Bruneton Sky");

	return 0;
}

void SkyClass::Draw(ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight & light)
{
	CallStart(drawSky);

	D3D11_MAPPED_SUBRESOURCE mappedResource;
	UINT stride = sizeof(XMFLOAT3);
	UINT offset = 0;

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

	mImmediateContext->RSSetState(mRastStateBasic);

	// PS
	cbPerFramePSType* dataPS;
	mImmediateContext->Map(cbPerFramePS, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	dataPS = (cbPerFramePSType*)mappedResource.pData;

	XMStoreFloat3(&(dataPS->gCameraPos), Camera->GetPositionRelSun());
	dataPS->gExposure = 0.5;
	dataPS->gSunDir = light.Direction();
	// change light direction to sun direction
	dataPS->gSunDir.x *= -1; dataPS->gSunDir.y *= -1; dataPS->gSunDir.z *= -1;

	mImmediateContext->Unmap(cbPerFramePS, 0);

	mImmediateContext->PSSetConstantBuffers(0, 1, &cbPerFramePS);

	mImmediateContext->PSSetShaderResources(0, 1, &inscatterSRV);
	mImmediateContext->PSSetShaderResources(1, 1, &transmitanceSRV);
	mImmediateContext->PSSetShaderResources(2, 1, &irradianceSRV);
	mImmediateContext->PSSetSamplers(0, 3, mSamplerStateBasic);
	mImmediateContext->PSSetShader(mPixelShaderToCube, NULL, 0);

	mImmediateContext->OMSetDepthStencilState(mDepthStencilStateSky, 0);
	
	mImmediateContext->DrawIndexed(6, 0, 0);

	mImmediateContext->OMSetDepthStencilState(0, 0);

	CallEnd(drawSky);
}

void SkyClass::DrawToCube(ID3D11DeviceContext1 * mImmediateContext, DirectionalLight & light)
{
	ViewportStack::Push(mImmediateContext, &cubeViewport);

	for (int i = 0; i < 6; ++i)
	{
		// cleare cube map face and depth buffer
		mImmediateContext->ClearRenderTargetView(cubeTextRTV[i], reinterpret_cast<const float*>(&DirectX::Colors::Silver));
		mImmediateContext->ClearDepthStencilView(cubeMapDSV, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);

		RenderTargetStack::Push(mImmediateContext, &cubeTextRTV[i], &cubeMapDSV);

		Draw(mImmediateContext, mCubeMapCamera[i], light);

		RenderTargetStack::Pop(mImmediateContext);
	}

	ViewportStack::Pop(mImmediateContext);
	
}

void SkyClass::DrawToMap(ID3D11DeviceContext1 * mImmediateContext, DirectionalLight & light)
{
	ID3D11UnorderedAccessView* ppUAViewNULL[2] = { NULL, NULL };
	ID3D11ShaderResourceView* ppSRVNULL[4] = { NULL, NULL, NULL, NULL };

	mImmediateContext->PSSetShaderResources(100, 1, ppSRVNULL);
	ViewportStack::Push(mImmediateContext, &cubeViewport);
	mImmediateContext->ClearRenderTargetView(mapRTV, reinterpret_cast<const float*>(&DirectX::Colors::Black));
	// todo: no depth buffer
	mImmediateContext->ClearDepthStencilView(cubeMapDSV, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0.0f);
	RenderTargetStack::Push(mImmediateContext, &mapRTV, &cubeMapDSV);
	
	UINT stride = sizeof(XMFLOAT3);
	UINT offset = 0;

	mImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	mImmediateContext->IASetVertexBuffers(0, 1, &mScreenQuadVB, &stride, &offset);
	mImmediateContext->IASetIndexBuffer(mScreenQuadIB, DXGI_FORMAT_R16_UINT, 0);
	mImmediateContext->IASetInputLayout(mInputLayout);

	// VS
	mImmediateContext->VSSetShader(mMapVertexShader, nullptr, 0);
	mImmediateContext->RSSetState(mRastStateBasic);

	// PS
	skyMapParams.sunDir = light.Direction();
	// change light direction to sun direction
	skyMapParams.sunDir.x *= -1; skyMapParams.sunDir.y *= -1; skyMapParams.sunDir.z *= -1;
	MapResources(mImmediateContext, skyMapCB, skyMapParams);

	mImmediateContext->PSSetConstantBuffers(0, 1, &skyMapCB);
	mImmediateContext->PSSetShader(mMapPixelShader, nullptr, 0);

	mImmediateContext->PSSetShaderResources(101, 1, &inscatterSRV);
	mImmediateContext->PSSetShaderResources(102, 1, &transmitanceSRV);
	mImmediateContext->PSSetShaderResources(103, 1, &deltaESRV);

	mImmediateContext->PSSetSamplers(0, 1, mSamplerStateBasic);

	mImmediateContext->OMSetDepthStencilState(mDepthStencilStateSky, 0);

	mImmediateContext->DrawIndexed(6, 0, 0);

	mImmediateContext->OMSetDepthStencilState(nullptr, 0);

	RenderTargetStack::Pop(mImmediateContext);
	ViewportStack::Pop(mImmediateContext);

	mImmediateContext->GenerateMips(mapSRV);
	mImmediateContext->PSSetShaderResources(100, 1, &mapSRV);
	mImmediateContext->PSSetSamplers(1, 1, &mSamplerAnisotropic);

	
}

void SkyClass::DrawToScreen(ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight& light)
{
	ID3D11UnorderedAccessView* ppUAViewNULL[2] = { NULL, NULL };
	ID3D11ShaderResourceView* ppSRVNULL[4] = { NULL, NULL, NULL, NULL };

	D3D11_MAPPED_SUBRESOURCE mappedResource;
	UINT stride = sizeof(XMFLOAT3);
	UINT offset = 0;

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

	mImmediateContext->RSSetState(mRastStateBasic);

	// PS
	cbPerFramePSType* dataPS;
	mImmediateContext->Map(cbPerFramePS, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	dataPS = (cbPerFramePSType*)mappedResource.pData;

	XMStoreFloat3(&(dataPS->gCameraPos), Camera->GetPositionRelSun());
	dataPS->gExposure = 0.5;
	dataPS->gSunDir = light.Direction();
	// change light direction to sun direction
	dataPS->gSunDir.x *= -1; dataPS->gSunDir.y *= -1; dataPS->gSunDir.z *= -1;

	mImmediateContext->Unmap(cbPerFramePS, 0);

	mImmediateContext->PSSetConstantBuffers(0, 1, &cbPerFramePS);

	mImmediateContext->PSSetSamplers(3, 1, &mSamplerStateTrilinear);
	mImmediateContext->PSSetSamplers(1, 1, mSamplerStateBasic);
	mImmediateContext->PSSetShaderResources(1, 1, &transmitanceSRV);
	mImmediateContext->PSSetShader(mPixelShaderToScreen, NULL, 0);

	mImmediateContext->PSSetShaderResources(3, 1, &cubeTextSRV);

	mImmediateContext->OMSetDepthStencilState(mDepthStencilStateSky, 0);

	mImmediateContext->DrawIndexed(6, 0, 0);

	mImmediateContext->OMSetDepthStencilState(0, 0);

	mImmediateContext->PSSetShaderResources(3, 1, ppSRVNULL);
}

ID3D11ShaderResourceView * SkyClass::getTransmittanceSRV()
{
	return transmitanceSRV;
}

int SkyClass::Precompute(ID3D11DeviceContext1 * mImmediateContext)
{
	ID3D11UnorderedAccessView* ppUAViewNULL[2] = { NULL, NULL };
	ID3D11ShaderResourceView* ppSRVNULL[4] = { NULL, NULL, NULL, NULL };

	mImmediateContext->CSSetSamplers(0, 4, mSamplerStateBasic);

	// line 1
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, &transmitanceUAV, NULL);
	mImmediateContext->CSSetShader(transmittanceCS, NULL, 0);

	mImmediateContext->Dispatch(TRANSMITTANCE_W / 16, TRANSMITTANCE_H / 16, 1);

	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, NULL);

	// line 2
	mImmediateContext->CSSetShaderResources(0, 1, &transmitanceSRV);
	mImmediateContext->CSSetSamplers(0, 1, mSamplerStateBasic);
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, &deltaEUAV, NULL);
	mImmediateContext->CSSetShader(irradiance1CS, NULL, 0);

	mImmediateContext->Dispatch(SKY_W / 16, SKY_H / 16, 1);

	mImmediateContext->CSSetShaderResources(0, 1, ppSRVNULL);
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, NULL);

	// line 3
	mImmediateContext->CSSetShaderResources(0, 1, &transmitanceSRV);
	mImmediateContext->CSSetUnorderedAccessViews(0, 2, deltaSUAV, NULL);
	mImmediateContext->CSSetShader(inscatter1CS, NULL, 0);

	mImmediateContext->Dispatch(RES_MU_S*RES_NU / 16, RES_MU / 16, RES_R);

	mImmediateContext->CSSetShaderResources(0, 1, ppSRVNULL);
	mImmediateContext->CSSetUnorderedAccessViews(0, 2, ppUAViewNULL, NULL);

	// line 4

	// line 5
	mImmediateContext->CSSetShaderResources(0, 2, deltaSSRV);
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, &inscatterUAV, NULL);
	mImmediateContext->CSSetShader(copyInscatter1CS, NULL, 0);

	mImmediateContext->Dispatch(RES_MU_S*RES_NU / 16, RES_MU / 16, RES_R);

	mImmediateContext->CSSetShaderResources(0, 2, ppSRVNULL);
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, NULL);

	for (int order = 2; order <= 4; order++)
	{
		D3D11_MAPPED_SUBRESOURCE mappedResource;
		cbNOrderType* dataPtr;

		mImmediateContext->Map(cbNOrder, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
		dataPtr = (cbNOrderType*)mappedResource.pData;

		dataPtr->order[0] = order;

		mImmediateContext->Unmap(cbNOrder, 0);

		// line 7
		mImmediateContext->CSSetShaderResources(0, 1, &transmitanceSRV);
		mImmediateContext->CSSetShaderResources(1, 1, &deltaESRV);
		mImmediateContext->CSSetShaderResources(2, 2, deltaSSRV);
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, &deltaJUAV, NULL);
		mImmediateContext->CSSetShader(inscatterSCS, NULL, 0);
		mImmediateContext->CSSetConstantBuffers(0, 1, &cbNOrder);

		mImmediateContext->Dispatch(RES_MU_S*RES_NU / 16, RES_MU / 16, RES_R);

		mImmediateContext->CSSetShaderResources(0, 4, ppSRVNULL);
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, NULL);

		// line 8
		mImmediateContext->CSSetShaderResources(0, 1, &transmitanceSRV);
		mImmediateContext->CSSetShaderResources(1, 2, deltaSSRV);
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, &deltaEUAV, NULL);
		mImmediateContext->CSSetShader(irradianceNCS, NULL, 0);
		mImmediateContext->CSSetConstantBuffers(0, 1, &cbNOrder);

		mImmediateContext->Dispatch(SKY_W / 16, SKY_H / 16, 1);

		mImmediateContext->CSSetShaderResources(0, 2, ppSRVNULL);
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, NULL);

		// line 9
		mImmediateContext->CSSetShaderResources(0, 1, &transmitanceSRV);
		mImmediateContext->CSSetShaderResources(1, 1, &deltaJSRV);
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, &deltaSUAV[0], NULL);
		mImmediateContext->CSSetShader(inscatterNCS, NULL, 0);

		mImmediateContext->Dispatch(RES_MU_S*RES_NU / 16, RES_MU / 16, RES_R);

		mImmediateContext->CSSetShaderResources(0, 2, ppSRVNULL);
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, NULL);

		// line 10
		mImmediateContext->CSSetShaderResources(0, 1, &deltaESRV);
		mImmediateContext->CSSetShaderResources(1, 1, &irradianceSRV);
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, &copyIrradianceUAV, NULL);
		mImmediateContext->CSSetShader(copyIrradianceCS, NULL, 0);

		mImmediateContext->Dispatch(SKY_W / 16, SKY_H / 16, 1);

		mImmediateContext->CSSetShaderResources(0, 2, ppSRVNULL);
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, NULL);

		mImmediateContext->CopyResource(irradianceText, copyIrradianceText);

		// line 11
		mImmediateContext->CSSetShaderResources(0, 1, &inscatterSRV);
		mImmediateContext->CSSetShaderResources(1, 1, deltaSSRV);
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, &copyInscatterUAV, NULL);
		mImmediateContext->CSSetShader(copyInscatterNCS, NULL, 0);

		mImmediateContext->Dispatch(RES_MU_S*RES_NU / 16, RES_MU / 16, RES_R);

		mImmediateContext->CSSetShaderResources(0, 2, ppSRVNULL);
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, NULL);

		mImmediateContext->CopyResource(inscatterText, copyInscatterText);
	}

	return 0;
}

void SkyClass::BuildCubeFaceCamera(float x = 0, float y = 6360.0001f, float z = 0)
{
	XMVECTOR center = XMVectorSet(x, y, z, 0);

	XMVECTOR targets[6] =
	{
		XMVectorSet(x + 1.0f, y, z, 0),
		XMVectorSet(x - 1.0f, y, z, 0),
		XMVectorSet(x, y + 1.0f, z, 0),
		XMVectorSet(x, y - 1.0f, z, 0),
		XMVectorSet(x, y, z + 1.0f, 0),
		XMVectorSet(x, y, z - 1.0f, 0)
	};

	XMVECTOR ups[6] =
	{
		XMVectorSet(0.0f, 1.0f, 0.0f, 0),
		XMVectorSet(0.0f, 1.0f, 0.0f, 0),
		XMVectorSet(0.0f, 0.0f, -1.0f, 0),
		XMVectorSet(0.0f, 0.0f, +1.0f, 0),
		XMVectorSet(0.0f, 1.0f, 0.0f, 0),
		XMVectorSet(0.0f, 1.0f, 0.0f, 0)
	};

	for (int i = 0; i < 6; ++i)
	{
		mCubeMapCamera[i] = std::make_shared<CameraClass>();
		mCubeMapCamera[i]->LookAt(center, targets[i], ups[i]);
		mCubeMapCamera[i]->SetLens(0.5f*XM_PI, 1.0f, 0.1f, 1000.0f);
	}
}
