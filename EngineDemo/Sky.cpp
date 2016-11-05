#include "Sky.h"
//#include "Shaders\Sky\resolutions.h"

#include "Utilities\CreateShader.h"
#include "Utilities\RenderViewTargetStack.h"
#include "Utilities\CreateBuffer.h"
#include "Utilities\MapResources.h"

// NOTE: here?
#include "ShadersManager.h"
#include "RenderStates.h"

#include <DirectXColors.h>

#include "Shaders\Sky2\Structures.hlsli"

SkyClass::SkyClass()
	: skyMapSize(512)
{
}

SkyClass::~SkyClass()
{
	ReleaseCOM(cbPerFramePS);
	ReleaseCOM(cbPerFrameVS);

	ReleaseCOM(mInputLayout);
	ReleaseCOM(mVertexShader);
	ReleaseCOM(mPixelShaderToCube);
	ReleaseCOM(mPixelShaderToScreen);

	// release only first (rest point to the same resource)
	delete[] mSamplerStateBasic;

	Shutdown();	
}

int SkyClass::Init(ID3D11Device1 * device, ID3D11DeviceContext1 * mImmediateContext)
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

	// precompute constant buffer
	CreateConstantBuffer(device, sizeof(nOrderType), nOrderCB, "nOrderCB");

	// sampler states
	mSamplerStateBasic = new ID3D11SamplerState*[4];
	for (size_t i = 0; i < 4; i++) mSamplerStateBasic[i] = RenderStates::Sampler::TriLinearClampSS;


	D3D11_BUFFER_DESC cbDesc = {};
	cbDesc.Usage = D3D11_USAGE_DYNAMIC;
	cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	cbDesc.MiscFlags = 0;
	cbDesc.StructureByteStride = 0;

	//Precompute(mImmediateContext);
	
	// render's buffers
	cbDesc.ByteWidth = sizeof(cbPerFrameVSType);
	
	device->CreateBuffer(&cbDesc, NULL, &cbPerFrameVS);

	cbDesc.ByteWidth = sizeof(cbPerFramePSType);

	device->CreateBuffer(&cbDesc, NULL, &cbPerFramePS);

	// full screen quad
	vector<XMFLOAT3> patchVertices(4);
	
	patchVertices[0] = XMFLOAT3(-1.0f, -1.0f, 0.0f);
	patchVertices[1] = XMFLOAT3(-1.0f, 1.0f, 0.0f);
	patchVertices[2] = XMFLOAT3(1.0f, 1.0f, 0.0f);
	patchVertices[3] = XMFLOAT3(1.0f, -1.0f, 0.0f);

	mScreenQuad.SetVertices(device, &patchVertices[0], patchVertices.size());

	vector<USHORT> indices(6);

	indices[0] = 0;
	indices[1] = 1;
	indices[2] = 2;
	indices[3] = 0;
	indices[4] = 2;
	indices[5] = 3;

	mScreenQuad.SetIndices(device, &indices[0], indices.size());

	vector<MeshBuffer::Subset> subsets;
	MeshBuffer::Subset sub;
	sub.Id = 0;
	sub.VertexStart = 0;
	sub.VertexCount = 4;
	sub.FaceStart = 0;
	sub.FaceCount = 2;
	subsets.push_back(sub);
	mScreenQuad.SetSubsetTable(subsets);

	CreatePSFromFile(L"..\\Debug\\Shaders\\Sky\\SkyToCubePS.cso", device, mPixelShaderToCube);

	CreatePSFromFile(L"..\\Debug\\Shaders\\Sky\\SkyToScreenPS.cso", device, mPixelShaderToScreen, "SkyToScreenPS - mPixelShaderToScreen");

	mPixelShaderPostFX = ShadersManager::Instance()->GetPS("Sky::SkyPostFX");

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

	mMapPixelShader = ShadersManager::Instance()->GetPS("Sky::SkyMapPS");

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
	text3descFile.Depth = RES_ALT;
	text3descFile.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	text3descFile.Height = RES_VZ;
	text3descFile.MipLevels = 1;
	text3descFile.MiscFlags = 0;
	text3descFile.Usage = D3D11_USAGE_IMMUTABLE;
	text3descFile.Width = RES_VS * RES_SZ;

	srvdescFile.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE3D;
	srvdescFile.Texture3D.MipLevels = 1;
	srvdescFile.Texture3D.MostDetailedMip = 0;

	data = new char[RES_ALT*RES_VZ*RES_VS*RES_SZ * sizeof(XMFLOAT4)];

	stream.open("inscatter.raw", std::ifstream::binary);
	if (stream.good())
	{
		stream.read(data, RES_ALT*RES_VZ*RES_VS*RES_SZ * sizeof(XMFLOAT4));
		stream.close();
	}

	subFile.pSysMem = data;
	subFile.SysMemPitch = RES_SZ * RES_VS * sizeof(XMFLOAT4);
	subFile.SysMemSlicePitch = subFile.SysMemPitch * RES_VZ;

	ID3D11Texture3D* text3dFile;
	device->CreateTexture3D(&text3descFile, &subFile, &text3dFile);
	device->CreateShaderResourceView(text3dFile, &srvdescFile, &inscatterFile);
	ReleaseCOM(text3dFile);
	delete[] data;

	return 0;
}

HRESULT SkyClass::Shutdown()
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

void SkyClass::Draw(ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight & light)
{
	//Precompute(mImmediateContext);
	CallStart(drawSky);

	ID3D11ShaderResourceView* ppSRVNULL[4] = { NULL, NULL, NULL, NULL };

	D3D11_MAPPED_SUBRESOURCE mappedResource;
	UINT stride = sizeof(XMFLOAT3);
	UINT offset = 0;

	mImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
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

	mImmediateContext->RSSetState(RenderStates::Rasterizer::DefaultRS);

	// PS
	cbPerFramePSType* dataPS;
	mImmediateContext->Map(cbPerFramePS, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	dataPS = (cbPerFramePSType*)mappedResource.pData;

	XMStoreFloat3(&(dataPS->gCameraPos), Camera->GetPositionRelSun());
	dataPS->gSunDir = light.Direction();
	// change light direction to sun direction
	dataPS->gSunDir.x *= -1; dataPS->gSunDir.y *= -1; dataPS->gSunDir.z *= -1;

	mImmediateContext->Unmap(cbPerFramePS, 0);

	mImmediateContext->PSSetConstantBuffers(0, 1, &cbPerFramePS);

	mImmediateContext->PSSetShaderResources(0, 1, &inscatterFile);
	mImmediateContext->PSSetShaderResources(1, 1, &transmittanceFile);
	mImmediateContext->PSSetShaderResources(2, 1, &irradianceFile);
	mImmediateContext->PSSetSamplers(0, 3, mSamplerStateBasic);
	mImmediateContext->PSSetShader(mPixelShaderToCube, NULL, 0);

	mImmediateContext->OMSetDepthStencilState(RenderStates::DepthStencil::NoWriteGreaterEqualDSS, 0);
	
	mScreenQuad.Draw(mImmediateContext);

	mImmediateContext->OMSetDepthStencilState(RenderStates::DepthStencil::DefaultDSS, 0);

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
	mImmediateContext->IASetInputLayout(mInputLayout);

	// VS
	mImmediateContext->VSSetShader(mMapVertexShader, nullptr, 0);
	mImmediateContext->RSSetState(RenderStates::Rasterizer::DefaultRS);

	// PS
	skyMapParams.sunDir = light.Direction();
	// change light direction to sun direction
	skyMapParams.sunDir.x *= -1; skyMapParams.sunDir.y *= -1; skyMapParams.sunDir.z *= -1;
	MapResources(mImmediateContext, skyMapCB, skyMapParams);

	mImmediateContext->PSSetConstantBuffers(0, 1, &skyMapCB);
	mImmediateContext->PSSetShader(mMapPixelShader, nullptr, 0);

	mImmediateContext->PSSetShaderResources(101, 1, &inscatterFile);
	mImmediateContext->PSSetShaderResources(102, 1, &transmittanceFile);
	mImmediateContext->PSSetShaderResources(103, 1, &irradianceFile);

	mImmediateContext->PSSetSamplers(0, 1, mSamplerStateBasic);

	mImmediateContext->OMSetDepthStencilState(RenderStates::DepthStencil::NoWriteGreaterEqualDSS, 0);

	mScreenQuad.Draw(mImmediateContext);

	mImmediateContext->OMSetDepthStencilState(RenderStates::DepthStencil::DefaultDSS, 0);

	RenderTargetStack::Pop(mImmediateContext);
	ViewportStack::Pop(mImmediateContext);

	mImmediateContext->GenerateMips(newMapText->GetSRV());
	mImmediateContext->PSSetShaderResources(100, 1, newMapText->GetAddressOfSRV());
	mImmediateContext->PSSetSamplers(1, 1, &RenderStates::Sampler::AnisotropicWrapSS);

	
}

void SkyClass::Process(ID3D11DeviceContext1 * mImmediateContext, std::unique_ptr<PostFX::Canvas> const & Canvas, std::shared_ptr<CameraClass> Camera, DirectionalLight & light)
{
	ID3D11UnorderedAccessView* ppUAViewNULL[2] = { NULL, NULL };
	ID3D11ShaderResourceView* ppSRViewNULL[4] = { nullptr, nullptr, nullptr, nullptr };

	mImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	mImmediateContext->IASetInputLayout(mInputLayout);

	// VS
	cbPerFrameVSParams.gViewInverse = XMMatrixInverse(nullptr, XMMatrixTranspose(Camera->GetViewRelSun()));
	cbPerFrameVSParams.gProjInverse = XMMatrixInverse(nullptr, Camera->GetProjTrans());
	MapResources(mImmediateContext, cbPerFrameVS, cbPerFrameVSParams);

	mImmediateContext->VSSetConstantBuffers(0, 1, &cbPerFrameVS);
	mImmediateContext->VSSetShader(mVertexShader, nullptr, 0);

	mImmediateContext->RSSetState(RenderStates::Rasterizer::DefaultRS);

	// PS
	XMStoreFloat3(&cbPerFramePSParams.gCameraPos, Camera->GetPositionRelSun());
	cbPerFramePSParams.gSunDir = light.Direction();
	// change light direction to sun direction
	cbPerFramePSParams.gSunDir.x *= -1; cbPerFramePSParams.gSunDir.y *= -1; cbPerFramePSParams.gSunDir.z *= -1;
	XMFLOAT4X4 projMatrix;
	XMStoreFloat4x4(&projMatrix, Camera->GetProjMatrix());
	cbPerFramePSParams.gProj = XMFLOAT4{ projMatrix._33, projMatrix._34,
										 projMatrix._43, projMatrix._44 };
	MapResources(mImmediateContext, cbPerFramePS, cbPerFramePSParams);

	mImmediateContext->PSSetConstantBuffers(0, 1, &cbPerFramePS);
	mImmediateContext->PSSetConstantBuffers(1, 1, &cbPerFrameVS);
	mImmediateContext->PSSetShader(mPixelShaderPostFX, nullptr, 0);

	mImmediateContext->PSSetShaderResources(0, 1, &inscatterFile);
	mImmediateContext->PSSetShaderResources(1, 1, &transmittanceFile);
	mImmediateContext->PSSetShaderResources(2, 1, &irradianceFile);
	mImmediateContext->PSSetSamplers(0, 3, mSamplerStateBasic);

	mImmediateContext->PSSetShaderResources(3, 1, Canvas->GetDepthStencilSRV());

	mImmediateContext->OMSetDepthStencilState(RenderStates::DepthStencil::NoWriteGreaterEqualDSS, 0);

	mScreenQuad.Draw(mImmediateContext);

	mImmediateContext->OMSetDepthStencilState(RenderStates::DepthStencil::DefaultDSS, 0);

	mImmediateContext->PSSetShaderResources(0, 3, ppSRViewNULL);
}

ID3D11ShaderResourceView * SkyClass::getTransmittanceSRV()
{
	// UnresolvedMergeConflict (implement)
	return nullptr;// transmitanceSRV;
}

HRESULT SkyClass::Precompute(ID3D11DeviceContext1 * mImmediateContext)
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

	return S_OK;
}
