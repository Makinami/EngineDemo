#include "Sky.h"
#include "Shaders\Sky\resolutions.h"

#include "Utilities\CreateShader.h"
#include "Utilities\RenderViewTargetStack.h"
#include "Utilities\CreateBuffer.h"
#include "Utilities\MapResources.h"

#include <DirectXColors.h>

SkyClass::SkyClass()
	: mRastStateBasic(nullptr),
	skyMapSize(512)
{
}

SkyClass::~SkyClass()
{
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
	// 2D
	// transmitance
	EXIT_ON_NULL(newTransmittanceText =
				 TextureFactory::CreateTexture(D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
											   DXGI_FORMAT_R32G32B32A32_FLOAT,
											   TRANSMITTANCE_W, TRANSMITTANCE_H));

	// deltaE
	EXIT_ON_NULL(newDeltaEText =
				 TextureFactory::CreateTexture(D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
											   DXGI_FORMAT_R32G32B32A32_FLOAT,
											   SKY_W, SKY_H));

	// irradiance
	EXIT_ON_NULL(newIrradainceText =
				 TextureFactory::CreateTexture(D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
											   DXGI_FORMAT_R32G32B32A32_FLOAT,
											   SKY_W, SKY_H));

	EXIT_ON_NULL(newCopyIrradianceText =
				 TextureFactory::CreateTexture(D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
											   DXGI_FORMAT_R32G32B32A32_FLOAT,
											   SKY_W, SKY_H));

	// 3D
	D3D11_TEXTURE3D_DESC text3Desc;
	text3Desc.Width = RES_MU_S*RES_NU;
	text3Desc.Height = RES_MU;
	text3Desc.Depth = RES_R;
	text3Desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	text3Desc.CPUAccessFlags = 0;
	text3Desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	text3Desc.MipLevels = 1;
	text3Desc.MiscFlags = 0;
	text3Desc.Usage = D3D11_USAGE_DEFAULT;

	// deltaS
	newDeltaSText.resize(2);
	EXIT_ON_NULL(newDeltaSText[0] =
				 TextureFactory::CreateTexture(text3Desc));
	EXIT_ON_NULL(newDeltaSText[1] =
				 TextureFactory::CreateTexture(text3Desc));
	
	deltaSSRV.resize(2);
	deltaSSRV[0] = newDeltaSText[0]->GetSRV();
	deltaSSRV[1] = newDeltaSText[1]->GetSRV();

	deltaSUAV.resize(2);
	deltaSUAV[0] = newDeltaSText[0]->GetUAV();
	deltaSUAV[1] = newDeltaSText[1]->GetUAV();
	
	// inscatter
	EXIT_ON_NULL(newInscatterText =
				 TextureFactory::CreateTexture(text3Desc));

	EXIT_ON_NULL(newCopyInscatterText =
	TextureFactory::CreateTexture(text3Desc));

	// deltaJ	
	EXIT_ON_NULL(newDeltaJText =
				 TextureFactory::CreateTexture(text3Desc));

	CreateCSFromFile(L"..\\Debug\\Shaders\\Sky\\transmittance.cso", device, transmittanceCS);
	CreateCSFromFile(L"..\\Debug\\Shaders\\Sky\\irradiance1.cso", device, irradiance1CS);
	CreateCSFromFile(L"..\\Debug\\Shaders\\Sky\\inscatter1.cso", device, inscatter1CS);
	CreateCSFromFile(L"..\\Debug\\Shaders\\Sky\\copyInscatter1.cso", device, copyInscatter1CS);
	CreateCSFromFile(L"..\\Debug\\Shaders\\Sky\\inscatterS.cso", device, inscatterSCS);
	CreateCSFromFile(L"..\\Debug\\Shaders\\Sky\\irradianceN.cso", device, irradianceNCS);
	CreateCSFromFile(L"..\\Debug\\Shaders\\Sky\\inscatterN.cso", device, inscatterNCS);
	CreateCSFromFile(L"..\\Debug\\Shaders\\Sky\\copyIrradiance.cso", device, copyIrradianceCS);
	CreateCSFromFile(L"..\\Debug\\Shaders\\Sky\\copyInscatterN.cso", device, copyInscatterNCS);
	CreateCSFromFile(L"..\\Debug\\Shaders\\Sky\\zeroIrradiance.cso", device, zeroIrradiance);

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

	/*
	 * Sky Map
	 */
	skyMapViewport.TopLeftX = 0.0f;
	skyMapViewport.TopLeftY = 0.0f;
	skyMapViewport.Width = (float)skyMapSize;
	skyMapViewport.Height = (float)skyMapSize;
	skyMapViewport.MinDepth = 0.0f;
	skyMapViewport.MaxDepth = 1.0f;

	D3D11_TEXTURE2D_DESC text2Desc;
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

	EXIT_ON_NULL(newMapText =
				 TextureFactory::CreateTexture(text2Desc));

	CreateVSFromFile(L"..\\Debug\\Shaders\\Sky\\SkyMapVS.cso", device, mMapVertexShader, "SkyMapVS - mMapVertexShader");

	CreatePSFromFile(L"..\\Debug\\Shaders\\Sky\\SkyMapPS.cso", device, mMapPixelShader, "SkyMapPS - mMapPixelShader");

	CreateConstantBuffer(device, sizeof(skyMapBufferType), skyMapCB, "skyMapCB");

	drawSky = Performance->ReserveName(L"Bruneton Sky");

	// TEXT FROM FILE
	char* data;
	float* fdata;
	std::ifstream stream;

	D3D11_TEXTURE2D_DESC text2descFile;
	text2descFile.Height = SKY_H;
	text2descFile.Width = SKY_W;
	text2descFile.ArraySize = 1;
	text2descFile.Usage = D3D11_USAGE_IMMUTABLE;
	text2descFile.SampleDesc = { 1, 0 };
	text2descFile.MipLevels = 1;
	text2descFile.MiscFlags = 0;
	text2descFile.CPUAccessFlags = 0;
	text2descFile.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	text2descFile.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;

	D3D11_SHADER_RESOURCE_VIEW_DESC srvdescFile;
	srvdescFile.Texture2D.MipLevels = 1;
	srvdescFile.Texture2D.MostDetailedMip = 0;
	srvdescFile.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	srvdescFile.Format = text2descFile.Format;

	ID3D11Texture2D* text2dFile;

	data = new char[SKY_H*SKY_W * sizeof(XMFLOAT4)];

	stream.open("irradiance.raw", std::ifstream::binary);
	if (stream.good())
	{
		for (int i = 0; i < SKY_W * SKY_H; ++i)
			stream.read(data + i * sizeof(XMFLOAT4), sizeof(XMFLOAT3));
		stream.close();
	}

	D3D11_SUBRESOURCE_DATA subFile;
	subFile.pSysMem = data;
	subFile.SysMemPitch = SKY_W * sizeof(XMFLOAT4);

	device->CreateTexture2D(&text2descFile, &subFile, &text2dFile);
	device->CreateShaderResourceView(text2dFile, &srvdescFile, &irradianceFile);
	ReleaseCOM(text2dFile);

	delete[] data;

	text2descFile.Height = TRANSMITTANCE_H;
	text2descFile.Width = TRANSMITTANCE_W;

	data = new char[TRANSMITTANCE_H*TRANSMITTANCE_W * sizeof(XMFLOAT4)];

	stream.open("transmittance.raw", std::ifstream::binary);
	if (stream.good())
	{
		for (int i = 0; i < TRANSMITTANCE_W * TRANSMITTANCE_H; ++i)
			stream.read(data + i * sizeof(XMFLOAT4), sizeof(XMFLOAT3));
		stream.close();
	}

	subFile.pSysMem = data;
	subFile.SysMemPitch = TRANSMITTANCE_W * sizeof(XMFLOAT4);

	device->CreateTexture2D(&text2descFile, &subFile, &text2dFile);
	device->CreateShaderResourceView(text2dFile, &srvdescFile, &transmittanceFile);
	ReleaseCOM(text2dFile);

	delete[] data;

	D3D11_TEXTURE3D_DESC text3descFile;
	text3descFile.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	text3descFile.CPUAccessFlags = 0;
	text3descFile.Depth = RES_R;
	text3descFile.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	text3descFile.Height = RES_MU;
	text3descFile.MipLevels = 1;
	text3descFile.MiscFlags = 0;
	text3descFile.Usage = D3D11_USAGE_IMMUTABLE;
	text3descFile.Width = RES_MU_S * RES_NU;

	srvdescFile.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE3D;
	srvdescFile.Texture3D.MipLevels = 1;
	srvdescFile.Texture3D.MostDetailedMip = 0;

	data = new char[RES_R*RES_MU*RES_MU_S*RES_NU * sizeof(XMFLOAT4)];

	stream.open("inscatter.raw", std::ifstream::binary);
	if (stream.good())
	{
		stream.read(data, RES_R*RES_MU*RES_MU_S*RES_NU * sizeof(XMFLOAT4));
		stream.close();
	}

	subFile.pSysMem = data;
	subFile.SysMemPitch = RES_MU_S * RES_NU * sizeof(XMFLOAT4);
	subFile.SysMemSlicePitch = subFile.SysMemPitch * RES_MU;

	ID3D11Texture3D* text3dFile;
	device->CreateTexture3D(&text3descFile, &subFile, &text3dFile);
	device->CreateShaderResourceView(text3dFile, &srvdescFile, &inscatterFile);
	ReleaseCOM(text3dFile);
	delete[] data;

	return 0;
}

void SkyClass::Draw(ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight & light)
{
	//Precompute(mImmediateContext);
	CallStart(drawSky);

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

	mImmediateContext->PSSetShaderResources(0, 1, &inscatterFile);
	mImmediateContext->PSSetShaderResources(9, 1, newInscatterText->GetAddressOfSRV());
	mImmediateContext->PSSetShaderResources(1, 1, &transmittanceFile);
	mImmediateContext->PSSetShaderResources(10, 1, newTransmittanceText->GetAddressOfSRV());
	mImmediateContext->PSSetShaderResources(2, 1, &irradianceFile);
	mImmediateContext->PSSetShaderResources(11, 1, newIrradainceText->GetAddressOfSRV());
	mImmediateContext->PSSetSamplers(0, 3, mSamplerStateBasic);
	mImmediateContext->PSSetShader(mPixelShaderToCube, NULL, 0);

	mImmediateContext->OMSetDepthStencilState(mDepthStencilStateSky, 0);
	
	mImmediateContext->DrawIndexed(6, 0, 0);

	mImmediateContext->OMSetDepthStencilState(0, 0);

	mImmediateContext->PSSetShaderResources(9, 3, ppSRVNULL);

	CallEnd(drawSky);
}

void SkyClass::DrawToMap(ID3D11DeviceContext1 * mImmediateContext, DirectionalLight & light)
{
	ID3D11UnorderedAccessView* ppUAViewNULL[2] = { NULL, NULL };
	ID3D11ShaderResourceView* ppSRVNULL[4] = { NULL, NULL, NULL, NULL };

	mImmediateContext->PSSetShaderResources(100, 1, ppSRVNULL);
	ViewportStack::Push(mImmediateContext, &skyMapViewport);
	mImmediateContext->ClearRenderTargetView(newMapText->GetRTV(), reinterpret_cast<const float*>(&DirectX::Colors::Black));
	RenderTargetStack::Push(mImmediateContext, newMapText->GetAddressOfRTV(), nullptr);
	
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

	mImmediateContext->PSSetShaderResources(101, 1, newInscatterText->GetAddressOfSRV());
	mImmediateContext->PSSetShaderResources(102, 1, newTransmittanceText->GetAddressOfSRV());
	mImmediateContext->PSSetShaderResources(103, 1, newDeltaEText->GetAddressOfSRV());

	mImmediateContext->PSSetSamplers(0, 1, mSamplerStateBasic);

	mImmediateContext->OMSetDepthStencilState(mDepthStencilStateSky, 0);

	mImmediateContext->DrawIndexed(6, 0, 0);

	mImmediateContext->OMSetDepthStencilState(nullptr, 0);

	RenderTargetStack::Pop(mImmediateContext);
	ViewportStack::Pop(mImmediateContext);

	mImmediateContext->GenerateMips(newMapText->GetSRV());
	mImmediateContext->PSSetShaderResources(100, 1, newMapText->GetAddressOfSRV());
	mImmediateContext->PSSetSamplers(1, 1, &mSamplerAnisotropic);

	
}

ID3D11ShaderResourceView * SkyClass::getTransmittanceSRV()
{
	// UnresolvedMergeConflict (implement)
	return nullptr;// transmitanceSRV;
}

int SkyClass::Precompute(ID3D11DeviceContext1 * mImmediateContext)
{
	ID3D11UnorderedAccessView* ppUAViewNULL[2] = { NULL, NULL };
	ID3D11ShaderResourceView* ppSRVNULL[4] = { NULL, NULL, NULL, NULL };

	mImmediateContext->CSSetSamplers(0, 4, mSamplerStateBasic);

	// line 1
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, newTransmittanceText->GetAddressOfUAV(), NULL);
	mImmediateContext->CSSetShader(transmittanceCS, NULL, 0);

	mImmediateContext->Dispatch(TRANSMITTANCE_W / 16, TRANSMITTANCE_H / 16, 1);

	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, NULL);

	// line 2
	mImmediateContext->CSSetShaderResources(0, 1, newTransmittanceText->GetAddressOfSRV());
	mImmediateContext->CSSetSamplers(0, 1, mSamplerStateBasic);
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, newDeltaEText->GetAddressOfUAV(), NULL);
	mImmediateContext->CSSetShader(irradiance1CS, NULL, 0);

	mImmediateContext->Dispatch(SKY_W / 16, SKY_H / 16, 1);

	mImmediateContext->CSSetShaderResources(0, 1, ppSRVNULL);
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, NULL);

	// line 3
	mImmediateContext->CSSetShaderResources(0, 1, newTransmittanceText->GetAddressOfSRV());
	mImmediateContext->CSSetUnorderedAccessViews(0, 2, deltaSUAV.data(), NULL);
	mImmediateContext->CSSetShader(inscatter1CS, NULL, 0);

	mImmediateContext->Dispatch(RES_MU_S*RES_NU / 16, RES_MU / 16, RES_R);

	mImmediateContext->CSSetShaderResources(0, 1, ppSRVNULL);
	mImmediateContext->CSSetUnorderedAccessViews(0, 2, ppUAViewNULL, NULL);

	// line 4
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, newDeltaEText->GetAddressOfUAV(), nullptr);
	mImmediateContext->CSSetShader(zeroIrradiance, nullptr, 0);

	mImmediateContext->Dispatch(SKY_W / 16, SKY_H / 16, 1);

	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);
	

	// line 5
	mImmediateContext->CSSetShaderResources(0, 2, deltaSSRV.data());
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, newInscatterText->GetAddressOfUAV(), NULL);
	mImmediateContext->CSSetShader(copyInscatter1CS, NULL, 0);

	mImmediateContext->Dispatch(RES_MU_S*RES_NU / 16, RES_MU / 16, RES_R);

	mImmediateContext->CSSetShaderResources(0, 2, ppSRVNULL);
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, NULL);

	// line 6
	for (int order = 2; order <= 2; order++)
	{
		D3D11_MAPPED_SUBRESOURCE mappedResource;
		cbNOrderType* dataPtr;

		mImmediateContext->Map(cbNOrder, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
		dataPtr = (cbNOrderType*)mappedResource.pData;

		dataPtr->order[0] = order;

		mImmediateContext->Unmap(cbNOrder, 0);

		// line 7
		mImmediateContext->CSSetShaderResources(0, 1, newTransmittanceText->GetAddressOfSRV());
		mImmediateContext->CSSetShaderResources(1, 1, newDeltaEText->GetAddressOfSRV());
		mImmediateContext->CSSetShaderResources(2, 2, deltaSSRV.data());
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, newDeltaJText->GetAddressOfUAV(), NULL);
		mImmediateContext->CSSetShader(inscatterSCS, NULL, 0);
		mImmediateContext->CSSetConstantBuffers(0, 1, &cbNOrder);

		mImmediateContext->Dispatch(RES_MU_S*RES_NU / 16, RES_MU / 16, RES_R);

		mImmediateContext->CSSetShaderResources(0, 4, ppSRVNULL);
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, NULL);

		// line 8
		mImmediateContext->CSSetShaderResources(0, 1, newTransmittanceText->GetAddressOfSRV());
		mImmediateContext->CSSetShaderResources(1, 2, deltaSSRV.data());
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, newDeltaEText->GetAddressOfUAV(), NULL);
		mImmediateContext->CSSetShader(irradianceNCS, NULL, 0);
		mImmediateContext->CSSetConstantBuffers(0, 1, &cbNOrder);

		mImmediateContext->Dispatch(SKY_W / 16, SKY_H / 16, 1);

		mImmediateContext->CSSetShaderResources(0, 3, ppSRVNULL);
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, NULL);

		// line 9
		mImmediateContext->CSSetShaderResources(0, 1, newTransmittanceText->GetAddressOfSRV());
		mImmediateContext->CSSetShaderResources(1, 1, newDeltaJText->GetAddressOfSRV());
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, &deltaSUAV[0], NULL);
		mImmediateContext->CSSetShader(inscatterNCS, NULL, 0);

		mImmediateContext->Dispatch(RES_MU_S*RES_NU / 16, RES_MU / 16, RES_R);

		mImmediateContext->CSSetShaderResources(0, 2, ppSRVNULL);
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, NULL);

		// line 10
		mImmediateContext->CSSetShaderResources(0, 1, newDeltaEText->GetAddressOfSRV());
		mImmediateContext->CSSetShaderResources(1, 1, newIrradainceText->GetAddressOfSRV());
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, newCopyIrradianceText->GetAddressOfUAV(), NULL);
		mImmediateContext->CSSetShader(copyIrradianceCS, NULL, 0);

		mImmediateContext->Dispatch(SKY_W / 16, SKY_H / 16, 1);

		mImmediateContext->CSSetShaderResources(0, 2, ppSRVNULL);
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, NULL);

		mImmediateContext->CopyResource(newIrradainceText->GetTexture(), newCopyIrradianceText->GetTexture());

		// line 11
		mImmediateContext->CSSetShaderResources(0, 1, newInscatterText->GetAddressOfSRV());
		mImmediateContext->CSSetShaderResources(1, 1, deltaSSRV.data());
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, newCopyInscatterText->GetAddressOfUAV(), NULL);
		mImmediateContext->CSSetShader(copyInscatterNCS, NULL, 0);

		mImmediateContext->Dispatch(RES_MU_S*RES_NU / 16, RES_MU / 16, RES_R);

		mImmediateContext->CSSetShaderResources(0, 2, ppSRVNULL);
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, NULL);

		mImmediateContext->CopyResource(newInscatterText->GetTexture(), newCopyInscatterText->GetTexture());
	}

	return 0;
}
