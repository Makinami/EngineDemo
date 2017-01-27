#include "Ocean.h"

#include <amp.h>

#include <random>

#include "DDSTextureLoader.h"

#include "Utilities\CreateShader.h"
#include "Utilities\CreateBuffer.h"
#include "Utilities\MapResources.h"

#include "ShaderManager.h"

#include "RenderStates.h"

using Microsoft::WRL::ComPtr;

void ExtractFrustrumPlanes(XMFLOAT4 planes[6], CXMMATRIX M);


#define EXIT_ON_FAILURE(fnc)  \
	{ \
		HRESULT result; \
		if (FAILED(result = fnc)) { \
			return result; \
		} \
	}

OceanClass::OceanClass() :
	spectrumSRV(nullptr),
	initFFTCS(nullptr),
	fftCS(nullptr),
	JacobianInjectCS(nullptr),
	dissipateTurbulanceCS(nullptr),
	perFrameCB(nullptr),
	turbulenceTextReadId(0),
	turbulenceTextWriteId(1),
	FFT_SIZE(256),
	GRID_SIZE{ 5488.0f, 392.0f, 28.0f, 2.0f },
	varianceRes(16),
	windSpeed(10.0f),
	waveAge(0.84f),
	cm(0.23f),
	km(370.0f),
	spectrumGain(1.0f),
	time(0),
	rand01(0.0f, 1.0f),
	screenWidth(1280),
	screenHeight(720),
	screenGridSize(2),
	screen(0),
	changed(true),
	initFFTCB(nullptr),
	wind_heading(0),
	wind_speed(10),
	max_depth(100)
{
	random_device rd;
	mt.seed(rd());
}


OceanClass::~OceanClass()
{
	Release();
}

HRESULT OceanClass::Init(ID3D11Device1 *& device, ID3D11DeviceContext1 *& mImmediateContext)
{
	EXIT_ON_FAILURE(CompileShadersAndInputLayout(device));

	EXIT_ON_FAILURE(CreateConstantBuffers(device));

	EXIT_ON_FAILURE(CreateDataResources(device));

	EXIT_ON_FAILURE(CreateSamplerRasterDepthStencilStates(device));

	//CreateGridMesh(device);d

	//CreateScreenMesh(device);

	CreateDiscMesh(device);

	ID3D11UnorderedAccessView* ppUAViewNULL[2] = { NULL, NULL };
	ID3D11Buffer* buffers[] = { constCB[2].Get(), perFrameCB.Get() };

	// variances
	mImmediateContext->CSSetShaderResources(0, 1, spectrumSRV.GetAddressOf());
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, varianceUAV.GetAddressOf(), nullptr);
	mImmediateContext->CSSetShader(slopeVarianceCS.Get(), nullptr, 0);
	mImmediateContext->CSSetConstantBuffers(0, 2, buffers);
	//mImmediateContext->CSSetShaderResources(8, 1, phaseSRV.GetAddressOf());
	mImmediateContext->CSSetUnorderedAccessViews(8, 1, newSpectrumUAV.GetAddressOf(), nullptr);

	//mImmediateContext->Dispatch(varianceRes / 16, varianceRes / 16, 16);

	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);

	oceanQuadTree.Init(device, -40960.0f, -40960.0f, 81920.0f, 81920.0f, 15, 12, XMFLOAT3(30.0, 15.0, 30.0));

	CreateDDSTextureFromFile(device, L"Textures\\FoamCombined.dds", nullptr, &FoamCombinedSRV);
	CreateDDSTextureFromFile(device, L"Textures\\FoamRamp.dds", nullptr, &FoamRampSRV);
	CreateDDSTextureFromFile(device, L"Textures\\ColourDepthRamp.dds", nullptr, &ColourDepthRampSRV);

	myBar = TwNewBar("Ocean");

	TwAddVarRW(myBar, "Wind speed", TW_TYPE_FLOAT, &wind_speed, " group=Wind  min=2.7 max=33.0 step=0.1 precision=1 ");
	TwAddVarRW(myBar, "Wind heading", TW_TYPE_FLOAT, &wind_heading, " group=Wind ");
	
	perFrameParams.maxDepth = 100.0;
	TwAddVarRW(myBar, "MaxDep", TW_TYPE_FLOAT, &perFrameParams.maxDepth, " label='Max depth' min=1");
	perFrameParams.seaColourAlpha = 0.25;
	TwAddVarRW(myBar, "SeaColourA", TW_TYPE_FLOAT, &perFrameParams.seaColourAlpha, " label='Sea Colour Alpha' min=0 max=1 step=0.01 ");

	perFrameParams.SeaFlag = true;
	TwAddVarRW(myBar, "SeaFlag", TW_TYPE_BOOL32, &perFrameParams.SeaFlag, " label='Sea Light' ");
	perFrameParams.SunFlag = true;
	TwAddVarRW(myBar, "SunFlag", TW_TYPE_BOOL32, &perFrameParams.SunFlag, " label='Sun Light' ");
	perFrameParams.SkyFlag = true;
	TwAddVarRW(myBar, "SkyFlag", TW_TYPE_BOOL32, &perFrameParams.SkyFlag, " label='Sky Light' ");

	perFrameParams.seaColour = XMFLOAT4{ 10.0f / 255.0f, 40.0F / 255.0f, 120.0f / 255.0f, 25.5f / 255.0f };
	TwAddVarRW(myBar, "Red", TW_TYPE_FLOAT, &perFrameParams.seaColourSSS.x, "");
	TwAddVarRW(myBar, "Green", TW_TYPE_FLOAT, &perFrameParams.seaColourSSS.y, "");
	TwAddVarRW(myBar, "Blue", TW_TYPE_FLOAT, &perFrameParams.seaColourSSS.z, "");

	perFrameParams.arcHDep = 1.5;
	TwAddVarRW(myBar, "ArcHeightDep", TW_TYPE_FLOAT, &perFrameParams.arcHDep, " label='Wave strech factor' step=0.01 ");

	perFrameParams.rgbExtinction = XMFLOAT3{ 7.0, 30.0, 70.0 };
	TwAddVarRW(myBar, "extR", TW_TYPE_FLOAT, &perFrameParams.rgbExtinction.x, " label='Ext Red' step=0.1 ");
	TwAddVarRW(myBar, "extG", TW_TYPE_FLOAT, &perFrameParams.rgbExtinction.y, " label='Ext Green' step=0.1 ");
	TwAddVarRW(myBar, "extB", TW_TYPE_FLOAT, &perFrameParams.rgbExtinction.z, " label='Ext Blue' step=0.1 ");
	return S_OK;
}

void OceanClass::Update(ID3D11DeviceContext1 *& mImmediateContext, float dt, DirectionalLight& light, std::shared_ptr<CameraClass> Camera)
{
	time += dt;

	float horizon = max(min(Camera->GetHorizon(), 1.1f), -0.1f);
	//horizon = 0.35;
	if (screen)
		indicesToRender = (int)((horizon + 0.1)*screenHeight / screenGridSize) * indicesPerRow;

	perFrameParams.dt = dt;
	perFrameParams.time = time;

	// TODO: dependend on real camera settings
	perFrameParams.coneAngle = 1.0f / (tan(XM_PIDIV4) * sqrt(1 + 1.6f*1.6f + 1.0f / tan(XM_PIDIV4)));
	XMStoreFloat4x4(&(perFrameParams.screenToCamMatrix), XMMatrixInverse(nullptr, Camera->GetProjTrans()));
	XMStoreFloat4x4(&(perFrameParams.camToWorldMatrix), XMMatrixInverse(nullptr, XMMatrixTranspose(Camera->GetViewMatrix())));
	XMStoreFloat4x4(&(perFrameParams.worldToScreenMatrix), Camera->GetViewProjTransMatrix());
	XMStoreFloat3(&(perFrameParams.camPos), Camera->GetPosition());
	perFrameParams.gridSize = XMFLOAT2(screenGridSize / (float)screenWidth, screenGridSize / (float)screenHeight);
	perFrameParams.sunDir = light.Direction();
	perFrameParams.sunDir.x *= -1; perFrameParams.sunDir.y *= -1; perFrameParams.sunDir.z *= -1;
	perFrameParams.lambdaJ = 2.5f;
	perFrameParams.lambdaV = 1.0f;
	perFrameParams.scale = 1000 * sqrt(13 * abs(cameraPos.y));
	perFrameParams.camLookAt = Camera->GetLookAt();
	//perFrameParams.wind = XMFLOAT2{ float(15.0f*cos(perFrameParams.time/5.0)), float(15.0f*sin(perFrameParams.time/5.0)) };
	//perFrameParams.wind = XMFLOAT2{ float(7.5f*(sin(perFrameParams.time / 10.0) + 1.0)), 0.0f };

	XMFLOAT2 windnew = XMFLOAT2{ wind_speed * sin(XMConvertToRadians(wind_heading)), wind_speed * cos(XMConvertToRadians(wind_heading)) };
	if (perFrameParams.wind.x != windnew.x || perFrameParams.wind.y != windnew.y)
	{
		perFrameParams.wind = windnew;
		changed = true;
	}

	XMFLOAT4X4 projMatrix;
	XMStoreFloat4x4(&projMatrix, Camera->GetProjMatrix());
	perFrameParams.gProj = XMFLOAT4{ projMatrix._33, projMatrix._34, projMatrix._43, projMatrix._44 };

	XMStoreFloat3(&cameraPos, Camera->GetPosition());
	BuildInstanceBuffer(mImmediateContext, Camera);

	/*instances[0].clear();
	instances[0].push_back(XMFLOAT4X4(
	20.0, 0.0, 0.0, 0.0,
	0.0, 1.0, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	3000.0, 0.0, 0.0, 1.0
	));*/

	oceanQuadTree.GenerateTree(mImmediateContext, Camera);

	D3D11_MAPPED_SUBRESOURCE mappedResources;
	mImmediateContext->Map(gridInstancesVB.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResources);
	//memcpy(mappedResources.pData, &instances[0][0], sizeof(instances[0][0]) * instances[0].size());
	mImmediateContext->Unmap(gridInstancesVB.Get(), 0);

	ID3D11Buffer* pp = perFrameCB.Get();
	MapResources(mImmediateContext, perFrameCB.Get(), perFrameParams);

	if (changed)
	{
		ID3D11UnorderedAccessView* ppUAViewNULL[2] = { NULL, NULL };
		ID3D11Buffer* buffers[] = { constCB[2].Get(), perFrameCB.Get() };

		mImmediateContext->CSSetConstantBuffers(0, 1, constCB[2].GetAddressOf());
		mImmediateContext->CSSetConstantBuffers(1, 1, perFrameCB.GetAddressOf());
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, newSpectrumUAV.GetAddressOf(), nullptr);

		mImmediateContext->CSSetShader(spectrumCS.Get(), nullptr, 0);

		mImmediateContext->Dispatch(16, 16, 4);

		// variances
		mImmediateContext->CSSetShaderResources(0, 1, spectrumSRV.GetAddressOf());
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, varianceUAV.GetAddressOf(), nullptr);
		mImmediateContext->CSSetShader(slopeVarianceCS.Get(), nullptr, 0);
		mImmediateContext->CSSetConstantBuffers(0, 2, buffers);
		mImmediateContext->CSSetShaderResources(8, 1, phaseSRV.GetAddressOf());
		mImmediateContext->CSSetUnorderedAccessViews(8, 1, newSpectrumUAV.GetAddressOf(), nullptr);

		mImmediateContext->Dispatch(varianceRes / 16, varianceRes / 16, 16);

		mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);

		changed = false;
	}

	//mImmediateContext->CSSetUnorderedAccessViews(15, 1, newSpectrumUAV.GetAddressOf(), nullptr);



	Simulate(mImmediateContext);
}

void OceanClass::Simulate(ID3D11DeviceContext1 *& mImmediateContext)
{
	ID3D11UnorderedAccessView* ppUAViewNULL[2] = { NULL, NULL };
	ID3D11ShaderResourceView* ppSRVNULL[2] = { NULL, NULL };

	ID3D11Buffer* buffers[] = { constCB[2].Get(), perFrameCB.Get() };

	// init
	// TODO: clean
	mImmediateContext->CSSetShader(initFFTCS.Get(), nullptr, 0);
	mImmediateContext->CSSetShaderResources(0, 1, spectrumSRV.GetAddressOf());
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, wavesUAV[0].GetAddressOf(), nullptr);
	mImmediateContext->CSSetShaderResources(8, 1, phaseSRV.GetAddressOf());
	mImmediateContext->CSSetUnorderedAccessViews(8, 1, newSpectrumUAV.GetAddressOf(), nullptr);
	mImmediateContext->CSSetConstantBuffers(0, 2, buffers);

	mImmediateContext->Dispatch(FFT_SIZE / 16, FFT_SIZE / 16, 1);

	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);
	mImmediateContext->CSSetUnorderedAccessViews(8, 1, ppUAViewNULL, nullptr);

	// fft
	mImmediateContext->CSSetShader(fftCS.Get(), nullptr, 0);
	for (auto i = 0; i < 2; ++i)
	{
		mImmediateContext->CSSetShaderResources(0, 1, wavesSRV[i].GetAddressOf());
		mImmediateContext->CSSetUnorderedAccessViews(0, 1, wavesUAV[1 - i].GetAddressOf(), nullptr);
		mImmediateContext->CSSetConstantBuffers(0, 1, constCB[i].GetAddressOf());

		mImmediateContext->Dispatch(1, 256, 6);

		mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);
	}

	// jacobian inject
	swap(turbulenceTextReadId, turbulenceTextWriteId);

	mImmediateContext->CSSetShader(JacobianInjectCS.Get(), nullptr, 0);
	mImmediateContext->CSSetShaderResources(0, 1, wavesSRV[0].GetAddressOf());
	mImmediateContext->CSSetShaderResources(1, 1, turbulenceSRV[turbulenceTextReadId].GetAddressOf());
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, turbulenceUAV[turbulenceTextWriteId].GetAddressOf(), nullptr);
	mImmediateContext->CSSetConstantBuffers(0, 2, buffers);

	mImmediateContext->Dispatch(FFT_SIZE / 16, FFT_SIZE / 16, 3);

	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);
	mImmediateContext->CSSetShaderResources(0, 1, ppSRVNULL);

	mImmediateContext->CSSetShader(nullptr, 0, 0);
}

void OceanClass::BuildInstanceBuffer(ID3D11DeviceContext1 *& mImmediateContext, std::shared_ptr<CameraClass> Camera)
{
	float edge = 1000 * sqrt(abs(13 * cameraPos.y));
	instances[0].clear();
	DivideTile(XMFLOAT2(-edge / 4.0f, edge / 4.0f), edge / 2.0f, 0);
	DivideTile(XMFLOAT2(edge / 4.0f, edge / 4.0f), edge / 2.0f, 1);
	DivideTile(XMFLOAT2(-edge / 4.0f, -edge / 4.0f), edge / 2.0f, 2);
	DivideTile(XMFLOAT2(edge / 4.0f, -edge / 4.0f), edge / 2.0f, 3);
}

void OceanClass::DivideTile(XMFLOAT2 pos, float edge, int num)
{

	float dist = sqrt((pos.x - cameraPos.x)*(pos.x - cameraPos.x) + cameraPos.y*cameraPos.y + (pos.y - cameraPos.z)*(pos.y - cameraPos.z));

	if (edge <= 100.0 || edge / sqrt(1.5) - dist < 0.0)
	{
		instances[0].push_back(XMFLOAT4X4(
			edge / 200.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, edge / 200.0f, 0.0f,
			pos.x, 0.0f, pos.y, 1.0f
		));
	}
	else
	{
		DivideTile(XMFLOAT2(pos.x - edge / 4.0f, pos.y + edge / 4.0f), edge / 2.0f, 0);
		DivideTile(XMFLOAT2(pos.x + edge / 4.0f, pos.y + edge / 4.0f), edge / 2.0f, 1);
		DivideTile(XMFLOAT2(pos.x - edge / 4.0f, pos.y - edge / 4.0f), edge / 2.0f, 2);
		DivideTile(XMFLOAT2(pos.x + edge / 4.0f, pos.y - edge / 4.0f), edge / 2.0f, 3);
	}
}

void OceanClass::Draw(ID3D11DeviceContext1 *& mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight& light)
{
	ID3D11ShaderResourceView* ppSRVNULL[] = { NULL, NULL, NULL };
	ID3D11Buffer* buffers[] = { constCB[2].Get(), perFrameCB.Get() };

	// IA 
	UINT stride[] = { sizeof(XMFLOAT2), sizeof(XMFLOAT4X4) };
	UINT offset[] = { 0, 0 };
	ID3D11Buffer* vbs[] = { gridMeshVB.Get(), gridInstancesVB.Get() };
	mImmediateContext->IASetInputLayout(mQuadIL.Get());
	mImmediateContext->IASetIndexBuffer(discMeshIB.Get(), DXGI_FORMAT_R16_UINT, 0);
	mImmediateContext->IASetVertexBuffers(0, 1, discMeshVB.GetAddressOf(), stride, offset);
	mImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST);

	// VS
	mImmediateContext->VSSetShader(mQuadVS.Get(), nullptr, 0);
	mImmediateContext->VSSetConstantBuffers(0, 2, buffers);
	mImmediateContext->VSSetShaderResources(0, 1, wavesSRV[0].GetAddressOf());
	mImmediateContext->VSSetSamplers(1, 1, &RenderStates::Sampler::AnisotropicWrapSS);

	mImmediateContext->VSSetShader(mVS2.Get(), nullptr, 0);

	// HS
	//mImmediateContext->HSSetShader(mHullShader.Get(), nullptr, 0);
	mImmediateContext->HSSetConstantBuffers(0, 2, buffers);

	mImmediateContext->HSSetShader(mHS2, nullptr, 0);
	mImmediateContext->HSSetShaderResources(0, 1, wavesSRV[0].GetAddressOf());
	mImmediateContext->HSSetSamplers(1, 1, &RenderStates::Sampler::AnisotropicWrapSS);

	// DS
	//mImmediateContext->DSSetShader(mDomainShader.Get(), nullptr, 0);
	mImmediateContext->DSSetConstantBuffers(0, 2, buffers);
	mImmediateContext->DSSetShaderResources(0, 1, wavesSRV[0].GetAddressOf());
	mImmediateContext->DSSetSamplers(1, 1, &RenderStates::Sampler::AnisotropicWrapSS);

	mImmediateContext->DSSetShader(mDS2, nullptr, 0);

	// PS 
	mImmediateContext->PSSetShader(mPixelShader.Get(), nullptr, 0);
	mImmediateContext->PSSetConstantBuffers(0, 2, buffers);
	mImmediateContext->PSSetShaderResources(0, 1, wavesSRV[0].GetAddressOf());
	mImmediateContext->PSSetSamplers(1, 1, &RenderStates::Sampler::AnisotropicWrapSS);
	mImmediateContext->PSSetShaderResources(1, 1, fresnelSRV.GetAddressOf());
	mImmediateContext->PSSetSamplers(2, 1, mSamplerClamp.GetAddressOf());
	mImmediateContext->PSSetShaderResources(2, 1, turbulenceSRV[turbulenceTextReadId].GetAddressOf());
	mImmediateContext->PSSetShaderResources(3, 1, noiseSRV.GetAddressOf());
	mImmediateContext->PSSetShaderResources(4, 1, varianceSRV.GetAddressOf());
	mImmediateContext->PSSetSamplers(3, 1, &RenderStates::Sampler::BilinearWrapSS);
	mImmediateContext->PSSetShaderResources(13, 1, foamSRV.GetAddressOf());

	mImmediateContext->PSSetShader(mPS2, nullptr, 0);

	// RS & OM
	//mImmediateContext->RSSetState(RenderStates::Rasterizer::WireframeRS);
	mImmediateContext->OMSetDepthStencilState(mDepthStencilState.Get(), 0);

	//mImmediateContext->DrawIndexed(indicesToRender, 0, 0);
	//mImmediateContext->DrawIndexedInstanced(indicesToRender, instances[0].size(), 0, 0, 0);

	oceanQuadTree.Draw(mImmediateContext);

	mImmediateContext->RSSetState(RenderStates::Rasterizer::DefaultRS);

	mImmediateContext->VSSetShaderResources(0, 1, ppSRVNULL);
	mImmediateContext->HSSetShader(nullptr, nullptr, 0);
	mImmediateContext->HSSetShaderResources(0, 1, ppSRVNULL);
	mImmediateContext->DSSetShader(nullptr, nullptr, 0);
	mImmediateContext->DSSetShaderResources(0, 1, ppSRVNULL);
	mImmediateContext->PSSetShaderResources(0, 3, ppSRVNULL);
}

void OceanClass::DrawPost(ID3D11DeviceContext1 *& mImmediateContext, std::unique_ptr<PostFX::Canvas> const & Canvas, std::shared_ptr<CameraClass> Camera, DirectionalLight & light)
{
	ID3D11ShaderResourceView* ppSRVNULL[] = { NULL, NULL, NULL };
	ID3D11Buffer* buffers[] = { constCB[2].Get(), perFrameCB.Get() };

	// IA 
	UINT stride[] = { sizeof(XMFLOAT2), sizeof(XMFLOAT4X4) };
	UINT offset[] = { 0, 0 };
	ID3D11Buffer* vbs[] = { gridMeshVB.Get(), gridInstancesVB.Get() };
	mImmediateContext->IASetInputLayout(mQuadIL.Get());
	mImmediateContext->IASetIndexBuffer(discMeshIB.Get(), DXGI_FORMAT_R16_UINT, 0);
	mImmediateContext->IASetVertexBuffers(0, 1, discMeshVB.GetAddressOf(), stride, offset);
	mImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST);

	// VS
	mImmediateContext->VSSetShader(mQuadVS.Get(), nullptr, 0);
	mImmediateContext->VSSetConstantBuffers(0, 2, buffers);
	mImmediateContext->VSSetShaderResources(0, 1, wavesSRV[0].GetAddressOf());
	mImmediateContext->VSSetSamplers(1, 1, &RenderStates::Sampler::BilinearClampSS);

	mImmediateContext->VSSetShader(mVS2.Get(), nullptr, 0);

	// HS
	//mImmediateContext->HSSetShader(mHullShader.Get(), nullptr, 0);
	mImmediateContext->HSSetConstantBuffers(0, 2, buffers);

	mImmediateContext->HSSetShader(mHS2, nullptr, 0);
	mImmediateContext->HSSetShaderResources(0, 1, wavesSRV[0].GetAddressOf());
	//mImmediateContext->HSSetSamplers(1, 1, &RenderStates::Sampler::AnisotropicWrapSS);

	// DS
	//mImmediateContext->DSSetShader(mDomainShader.Get(), nullptr, 0);
	mImmediateContext->DSSetConstantBuffers(0, 2, buffers);
	mImmediateContext->DSSetShaderResources(0, 1, wavesSRV[0].GetAddressOf());
	mImmediateContext->DSSetSamplers(1, 1, &RenderStates::Sampler::BilinearWrapSS);

	mImmediateContext->DSSetShader(mDS2, nullptr, 0);

	// PS 
	mImmediateContext->PSSetShader(mPixelShader.Get(), nullptr, 0);
	mImmediateContext->PSSetConstantBuffers(0, 2, buffers);
	mImmediateContext->PSSetShaderResources(0, 1, wavesSRV[0].GetAddressOf());
	mImmediateContext->PSSetSamplers(4, 1, &RenderStates::Sampler::BilinearWrapSS);
	mImmediateContext->PSSetSamplers(1, 1, &RenderStates::Sampler::AnisotropicWrapSS);
	mImmediateContext->PSSetShaderResources(1, 1, fresnelSRV.GetAddressOf());
	mImmediateContext->PSSetSamplers(2, 1, mSamplerClamp.GetAddressOf());
	mImmediateContext->PSSetShaderResources(2, 1, turbulenceSRV[turbulenceTextReadId].GetAddressOf());
	mImmediateContext->PSSetShaderResources(3, 1, noiseSRV.GetAddressOf());
	mImmediateContext->PSSetShaderResources(4, 1, varianceSRV.GetAddressOf());
	mImmediateContext->PSSetSamplers(3, 1, &RenderStates::Sampler::BilinearWrapSS);
	mImmediateContext->PSSetSamplers(10, 1, &RenderStates::Sampler::BilinearClampSS);
	mImmediateContext->PSSetShaderResources(13, 1, foamSRV.GetAddressOf());
	mImmediateContext->PSSetShaderResources(20, 1, FoamCombinedSRV.GetAddressOf());
	mImmediateContext->PSSetShaderResources(21, 1, FoamRampSRV.GetAddressOf());
	mImmediateContext->PSSetShaderResources(22, 1, ColourDepthRampSRV.GetAddressOf());

	mImmediateContext->PSSetShaderResources(14, 1, Canvas->GetDepthCopySRV());
	mImmediateContext->PSSetShaderResources(15, 1, Canvas->GetAddressOfSRV(true));

	mImmediateContext->PSSetShader(mPS3, nullptr, 0);

	// RS & OM
	//mImmediateContext->RSSetState(RenderStates::Rasterizer::WireframeRS);
	//mImmediateContext->OMSetDepthStencilState(mDepthStencilState.Get(), 0);

	//mImmediateContext->DrawIndexed(indicesToRender, 0, 0);
	//mImmediateContext->DrawIndexedInstanced(indicesToRender, instances[0].size(), 0, 0, 0);

	//mImmediateContext->RSSetState(RenderStates::Rasterizer::WireframeRS);

	oceanQuadTree.Draw(mImmediateContext);

	//mImmediateContext->RSSetState(RenderStates::Rasterizer::DefaultRS);

	mImmediateContext->VSSetShaderResources(0, 1, ppSRVNULL);
	mImmediateContext->HSSetShader(nullptr, nullptr, 0);
	mImmediateContext->HSSetShaderResources(0, 1, ppSRVNULL);
	mImmediateContext->DSSetShader(nullptr, nullptr, 0);
	mImmediateContext->DSSetShaderResources(0, 1, ppSRVNULL);
	mImmediateContext->PSSetShaderResources(0, 3, ppSRVNULL);
	mImmediateContext->PSSetShaderResources(14, 2, ppSRVNULL);
}

void OceanClass::Release()
{
	ReleaseCOM(initFFTCB);

	if (myBar)
	{
		TwDeleteBar(myBar);
		myBar = nullptr;
	}
}

HRESULT OceanClass::CompileShadersAndInputLayout(ID3D11Device1 *& device)
{
	EXIT_ON_FAILURE(CreateCSFromFile(L"..\\Debug\\Shaders\\WaterBruneton\\variances.cso", device, slopeVarianceCS));

	EXIT_ON_FAILURE(CreateCSFromFile(L"..\\Debug\\Shaders\\Ocean\\initFFT.cso", device, initFFTCS));

	EXIT_ON_FAILURE(CreateCSFromFile(L"..\\Debug\\Shaders\\WaterBruneton\\FFT.cso", device, fftCS));

	EXIT_ON_FAILURE(CreateCSFromFile(L"..\\Debug\\Shaders\\Ocean\\JacobianInject.cso", device, JacobianInjectCS));

	EXIT_ON_FAILURE(CreateCSFromFile(L"..\\Debug\\Shaders\\Ocean\\dissipateTrubulence.cso", device, dissipateTurbulanceCS));

	EXIT_ON_FAILURE(CreatePSFromFile(L"..\\Debug\\Shaders\\Ocean\\OceanPS.cso", device, mPixelShader));

	EXIT_ON_FAILURE(CreateDSFromFile(L"..\\Debug\\Shaders\\Ocean\\OceanDS.cso", device, mDomainShader));

	EXIT_ON_FAILURE(CreateDSFromFile(L"..\\Debug\\Shaders\\Ocean\\OceanDSGerstner.cso", device, mGerstnerDS));

	EXIT_ON_FAILURE(CreateHSFromFile(L"..\\Debug\\Shaders\\Ocean\\OceanHS.cso", device, mHullShader));

	// vertex and input layout
	const D3D11_INPUT_ELEMENT_DESC vertexDesc[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 }
	};

	UINT numElements = sizeof(vertexDesc) / sizeof(vertexDesc[0]);

	EXIT_ON_FAILURE(CreateVSAndInputLayout(L"..\\Debug\\Shaders\\Ocean\\OceanVS.cso", device, mVertexShader, vertexDesc, numElements, mInputLayout));

	// vs & il quad
	const D3D11_INPUT_ELEMENT_DESC vertexQuadDesc[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "PSIZE", 0, DXGI_FORMAT_R32_FLOAT, 1, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
		{ "PSIZE", 1, DXGI_FORMAT_R32_FLOAT, 1, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
		{ "BLENDINDICES", 0, DXGI_FORMAT_R32_UINT, 1, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
		{ "PSIZE", 2, DXGI_FORMAT_R32_FLOAT, 1, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_INSTANCE_DATA, 1}
	};

	numElements = sizeof(vertexQuadDesc) / sizeof(vertexQuadDesc[0]);

	EXIT_ON_FAILURE(CreateVSAndInputLayout(L"..\\Debug\\Shaders\\Ocean\\OceanQuadVS.cso", device, mQuadVS, vertexQuadDesc, numElements, mQuadIL));
	
	EXIT_ON_FAILURE(CreateVSAndInputLayout(L"..\\Debug\\Shaders\\Ocean\\OceanVS2.cso", device, mVS2, vertexQuadDesc, numElements, mQuadIL));
	mPS2 = ShaderManager::Instance()->GetPS("Ocean::OceanPS2");
	mHS2 = ShaderManager::Instance()->GetHS("Ocean::OceanHS2");
	mDS2 = ShaderManager::Instance()->GetDS("Ocean::OceanDS2");

	mPS3 = ShaderManager::Instance()->GetPS("Ocean::OceanPS3");

	spectrumCS = ShaderManager::Instance()->GetCS("Ocean::SpectrumCS");
	newInit = ShaderManager::Instance()->GetCS("Ocean::newInitFFTCS");
	
	return S_OK;
}

HRESULT OceanClass::CreateConstantBuffers(ID3D11Device1 *& device)
{
	D3D11_BUFFER_DESC constantBufferDesc;
	constantBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	constantBufferDesc.CPUAccessFlags = 0;
	constantBufferDesc.MiscFlags = 0;
	constantBufferDesc.StructureByteStride = 0;
	constantBufferDesc.Usage = D3D11_USAGE_IMMUTABLE;

	UINT buffer[4] = { 0, 0, 0, 0 };
	constantBufferDesc.ByteWidth = sizeof(buffer);

	D3D11_SUBRESOURCE_DATA initData;
	initData.pSysMem = buffer;

	// fft 0/row pass
	constCB.push_back(nullptr);
	EXIT_ON_FAILURE(device->CreateBuffer(&constantBufferDesc, &initData, &(constCB.back())));

	// fft 1/col pass
	buffer[0] = 1;
	constCB.push_back(nullptr);
	EXIT_ON_FAILURE(device->CreateBuffer(&constantBufferDesc, &initData, &(constCB.back())));

	// invert grid size
	float bufferf[12] = { XM_2PI / GRID_SIZE[0], XM_2PI / GRID_SIZE[1], XM_2PI / GRID_SIZE[2], XM_2PI / GRID_SIZE[3],
		GRID_SIZE[0], GRID_SIZE[1], GRID_SIZE[2], GRID_SIZE[3], XM_PI / GRID_SIZE[0], XM_PI * FFT_SIZE / GRID_SIZE[0], XM_PI * FFT_SIZE / GRID_SIZE[1], XM_PI * FFT_SIZE / GRID_SIZE[2] };
	constantBufferDesc.ByteWidth = sizeof(bufferf);
	initData.pSysMem = bufferf;

	constCB.push_back(nullptr);
	EXIT_ON_FAILURE(device->CreateBuffer(&constantBufferDesc, &initData, &(constCB.back())));

	// per frame params
	EXIT_ON_FAILURE(CreateConstantBuffer(device, sizeof(perFrameParams), perFrameCB, "ocean per frame params __FIE__ __LINE__"));

	return S_OK;
}

HRESULT OceanClass::CreateDataResources(ID3D11Device1 *& device)
{
	/*
	* SPECTRUM
	*/
	UINT SLICE_SIZE = FFT_SIZE * FFT_SIZE * 2;
	float* spectrum = new float[SLICE_SIZE * 4];

	// populate wave spectrum array
	for (int y = 0; y < FFT_SIZE; ++y)
	{
		for (int x = 0; x < FFT_SIZE; ++x)
		{
			int offset = 2 * (y * FFT_SIZE + x);
			// [-N/2;N/2]; F[n] == F[N-n]
			int i = x >= FFT_SIZE / 2 ? x - FFT_SIZE : x;
			int j = y >= FFT_SIZE / 2 ? y - FFT_SIZE : y;

			// populate spectrum
			getSpectrumSample(i, j, GRID_SIZE[0], XM_PI / GRID_SIZE[0], spectrum + offset);
			getSpectrumSample(i, j, GRID_SIZE[1], XM_PI * FFT_SIZE / GRID_SIZE[0], spectrum + SLICE_SIZE + offset);
			getSpectrumSample(i, j, GRID_SIZE[2], XM_PI * FFT_SIZE / GRID_SIZE[1], spectrum + 2 * SLICE_SIZE + offset);
			getSpectrumSample(i, j, GRID_SIZE[3], XM_PI * FFT_SIZE / GRID_SIZE[2], spectrum + 3 * SLICE_SIZE + offset);
		}
	}

	D3D11_TEXTURE2D_DESC textDesc;
	textDesc.ArraySize = 4;
	textDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	textDesc.CPUAccessFlags = 0;
	textDesc.Format = DXGI_FORMAT_R32G32_FLOAT;
	textDesc.Height = FFT_SIZE;
	textDesc.MipLevels = 1;
	textDesc.MiscFlags = 0;
	textDesc.SampleDesc = { 1, 0 };
	textDesc.Usage = D3D11_USAGE_IMMUTABLE;
	textDesc.Width = FFT_SIZE;

	D3D11_SUBRESOURCE_DATA textData[4];
	for (auto i = 0; i < 4; ++i)
	{
		textData[i].pSysMem = spectrum + i * SLICE_SIZE;
		textData[i].SysMemPitch = sizeof(float) * FFT_SIZE * 2;
	}

	ComPtr<ID3D11Texture2D> spectrumText;
	EXIT_ON_FAILURE(device->CreateTexture2D(&textDesc, textData, &spectrumText));

	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
	srvDesc.Format = textDesc.Format;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
	srvDesc.Texture2DArray.ArraySize = textDesc.ArraySize;
	srvDesc.Texture2DArray.FirstArraySlice = 0;
	srvDesc.Texture2DArray.MipLevels = textDesc.MipLevels;
	srvDesc.Texture2DArray.MostDetailedMip = 0;

	EXIT_ON_FAILURE(device->CreateShaderResourceView(spectrumText.Get(), &srvDesc, &spectrumSRV));

	// before deleting local copy, calculate gather total slope variance
	float totalSlopeVariance = 0.0;
	for (int y = 0; y < FFT_SIZE; ++y)
	{
		for (int x = 0; x < FFT_SIZE; ++x)
		{
			int offset = 2 * (y * FFT_SIZE + x);
			// F[n] == F[N-n]
			float i = XM_2PI * (x >= FFT_SIZE / 2 ? x - FFT_SIZE : x);
			float j = XM_2PI * (y >= FFT_SIZE / 2 ? y - FFT_SIZE : y);

			// sum slope variance
			totalSlopeVariance += getSlopeVariance(i / GRID_SIZE[0], j / GRID_SIZE[0], spectrum + offset);
			totalSlopeVariance += getSlopeVariance(i / GRID_SIZE[1], j / GRID_SIZE[1], spectrum + SLICE_SIZE + offset);
			totalSlopeVariance += getSlopeVariance(i / GRID_SIZE[2], j / GRID_SIZE[2], spectrum + 2 * SLICE_SIZE + offset);
			totalSlopeVariance += getSlopeVariance(i / GRID_SIZE[3], j / GRID_SIZE[3], spectrum + 3 * SLICE_SIZE + offset);
		}
	}

	// theoretical slope variance
	float theoreticalSlopeVariance = 0.0f;
	float k = 5e-3f;
	while (k < 1e3f)
	{
		float nextK = k * 1.001;
		theoreticalSlopeVariance += k * k * this->spectrum(k, 0, true) * (nextK - k);
		k = nextK;
	}

	perFrameParams.mipmapLevel = 0.5f*(theoreticalSlopeVariance - totalSlopeVariance);

	delete[] spectrum;

	/*
	* FFT TEXTURES
	*/
	// TODO: Check how about 32bit for init and first FFT and 16bit only for output for rendering
	// And make sure is it fine to use 16bit for other textures
	textDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	textDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
	textDesc.MipLevels = 1;
	textDesc.Usage = D3D11_USAGE_DEFAULT;
	textDesc.ArraySize = 6;

	vector< ComPtr<ID3D11Texture2D> > fftText;
	for (auto i = 0; i < 2; ++i)
	{
		fftText.push_back(nullptr);
		EXIT_ON_FAILURE(device->CreateTexture2D(&textDesc, nullptr, &(fftText.back())));
	}

	srvDesc.Format = textDesc.Format;
	srvDesc.Texture2DArray.MipLevels = -1;
	srvDesc.Texture2DArray.ArraySize = textDesc.ArraySize;

	for (auto i = 0; i < 2; ++i)
	{
		wavesSRV.push_back(nullptr);
		EXIT_ON_FAILURE(device->CreateShaderResourceView(fftText[i].Get(), &srvDesc, &(wavesSRV.back())));
	}

	D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc;
	uavDesc.Format = textDesc.Format;
	uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2DARRAY;
	uavDesc.Texture2DArray.ArraySize = textDesc.ArraySize;
	uavDesc.Texture2DArray.FirstArraySlice = 0;
	uavDesc.Texture2DArray.MipSlice = 0;

	for (auto i = 0; i < 2; ++i)
	{
		wavesUAV.push_back(nullptr);
		EXIT_ON_FAILURE(device->CreateUnorderedAccessView(fftText[i].Get(), &uavDesc, &(wavesUAV.back())));
	}

	/*
	* TURBULENCE TEXTURES
	*/
	textDesc.ArraySize = 4;
	srvDesc.Texture2DArray.ArraySize = textDesc.ArraySize;
	uavDesc.Texture2DArray.ArraySize = textDesc.ArraySize;

	vector< ComPtr<ID3D11Texture2D> >turbulenceText;
	for (auto i = 0; i < 2; ++i)
	{
		turbulenceText.push_back(nullptr);
		EXIT_ON_FAILURE(device->CreateTexture2D(&textDesc, nullptr, &(turbulenceText.back())));

		turbulenceSRV.push_back(nullptr);
		EXIT_ON_FAILURE(device->CreateShaderResourceView(turbulenceText[i].Get(), &srvDesc, &(turbulenceSRV.back())));

		turbulenceUAV.push_back(nullptr);
		EXIT_ON_FAILURE(device->CreateUnorderedAccessView(turbulenceText[i].Get(), &uavDesc, &(turbulenceUAV.back())));
	}

	/*
	* FRESNEL TERM LOOKUP TEXTURE
	*/
	const int fresnelRes = 128;
	float* fresnel = new float[fresnelRes * 2];
	for (int i = 0; i < fresnelRes; ++i)
	{
		fresnel[i] = Fresnel(i / (float)(fresnelRes - 1), 1.0f, 1.333f);
		fresnel[i + fresnelRes] = Fresnel(i / (float)(fresnelRes - 1), 1.333f, 1.0f);
	}

	D3D11_TEXTURE1D_DESC text1Desc;
	text1Desc.ArraySize = 2;
	text1Desc.BindFlags = D3D10_BIND_SHADER_RESOURCE;
	text1Desc.CPUAccessFlags = 0;
	text1Desc.Format = DXGI_FORMAT_R32_FLOAT;
	text1Desc.MipLevels = 1;
	text1Desc.MiscFlags = 0;
	text1Desc.Usage = D3D11_USAGE_IMMUTABLE;
	text1Desc.Width = fresnelRes;

	for (auto i : { 0, 1 })
	{
		textData[i].pSysMem = fresnel + fresnelRes * i;
		textData[i].SysMemPitch = sizeof(float) * fresnelRes;
	}

	ComPtr<ID3D11Texture1D> fresnelText;
	EXIT_ON_FAILURE(device->CreateTexture1D(&text1Desc, textData, &fresnelText));

	srvDesc.Format = text1Desc.Format;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE1DARRAY;
	srvDesc.Texture1DArray.ArraySize = text1Desc.ArraySize;
	srvDesc.Texture1DArray.FirstArraySlice = 0;
	srvDesc.Texture1DArray.MipLevels = 1;
	srvDesc.Texture1DArray.MostDetailedMip = 0;

	EXIT_ON_FAILURE(device->CreateShaderResourceView(fresnelText.Get(), &srvDesc, &fresnelSRV));

	delete[] fresnel;

	/*
	* SLOPE VARIANCE TEXTURE
	*/
	float sigma2 = 0.01f * (windSpeed <= 7 ? 0.9f + 1.2f*log(windSpeed) : -8.4f + 6.0f*log(windSpeed));
	sigma2 = max(sigma2, 2e-5f);
	perFrameParams.sigma2 = XMFLOAT2(10.0f / 9.0f * sigma2, 8.0f / 9.0f * sigma2);

	D3D11_TEXTURE3D_DESC textDesc3D;

	textDesc3D.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	textDesc3D.CPUAccessFlags = 0;
	textDesc3D.Format = DXGI_FORMAT_R16G16_FLOAT;
	textDesc3D.Height = 16;
	textDesc3D.MipLevels = 1;
	textDesc3D.MiscFlags = 0;
	textDesc3D.Usage = D3D11_USAGE_DEFAULT;
	textDesc3D.Width = 16;
	textDesc3D.Depth = 16;

	ComPtr<ID3D11Texture3D> varianceText;
	EXIT_ON_FAILURE(device->CreateTexture3D(&textDesc3D, nullptr, &varianceText));

	srvDesc.Format = textDesc3D.Format;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE3D;
	srvDesc.Texture3D.MipLevels = 1;
	srvDesc.Texture3D.MostDetailedMip = 0;

	EXIT_ON_FAILURE(device->CreateShaderResourceView(varianceText.Get(), &srvDesc, &varianceSRV));

	uavDesc.Format = textDesc3D.Format;
	uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE3D;
	uavDesc.Texture3D.FirstWSlice = 0;
	uavDesc.Texture3D.MipSlice = 0;
	uavDesc.Texture3D.WSize = textDesc3D.Depth;

	EXIT_ON_FAILURE(device->CreateUnorderedAccessView(varianceText.Get(), &uavDesc, &varianceUAV));

	/*
	* INSTANCE BUFFER
	*/
	D3D11_BUFFER_DESC instanceDesc;
	instanceDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	instanceDesc.ByteWidth = sizeof(TileData) * 160;
	instanceDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	instanceDesc.MiscFlags = 0;
	instanceDesc.StructureByteStride = 0;
	instanceDesc.Usage = D3D11_USAGE_DYNAMIC;

	EXIT_ON_FAILURE(device->CreateBuffer(&instanceDesc, nullptr, &gridInstancesVB));

	instanceData.reserve(160);

	/*
	* NOISE
	*/
	CreateDDSTextureFromFile(device, L"Textures/noise.dds", NULL, &noiseSRV);

	/*
	* FOAM
	*/
	CreateDDSTextureFromFile(device, L"Textures/foam.dds", nullptr, &foamSRV);

	/*
	* PHASE
	*/
	std::default_random_engine rd;
	std::uniform_real_distribution<> dis(0.0, XM_2PI);
	std::function<float()> rnd = std::bind(dis, rd);

	std::vector<float> phase(4*FFT_SIZE*FFT_SIZE);
	for (auto&& i : phase)
		i = rnd();

	textDesc.ArraySize = 4;
	textDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	textDesc.CPUAccessFlags = 0;
	textDesc.Format = DXGI_FORMAT_R32_FLOAT;
	textDesc.Height = FFT_SIZE;
	textDesc.MipLevels = 1;
	textDesc.MiscFlags = 0;
	textDesc.SampleDesc = { 1, 0 };
	textDesc.Usage = D3D11_USAGE_IMMUTABLE;
	textDesc.Width = FFT_SIZE;

	for (auto i = 0; i < 4; ++i)
	{
		textData[i].pSysMem = &phase[i * FFT_SIZE * FFT_SIZE];
		textData[i].SysMemPitch = sizeof(float) * FFT_SIZE;
	}

	ComPtr<ID3D11Texture2D> phaseText;
	EXIT_ON_FAILURE(device->CreateTexture2D(&textDesc, textData, &phaseText));

	srvDesc.Format = textDesc.Format;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
	srvDesc.Texture2DArray.ArraySize = textDesc.ArraySize;
	srvDesc.Texture2DArray.FirstArraySlice = 0;
	srvDesc.Texture2DArray.MipLevels = 1;
	srvDesc.Texture2DArray.MostDetailedMip = 0;

	EXIT_ON_FAILURE(device->CreateShaderResourceView(phaseText.Get(), &srvDesc, &phaseSRV));

	textDesc.Format = DXGI_FORMAT_R32_FLOAT;
	textDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
	textDesc.Usage = D3D11_USAGE_DEFAULT;

	uavDesc.Format = textDesc.Format;
	uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2DARRAY;
	uavDesc.Texture2DArray.ArraySize = textDesc.ArraySize;
	uavDesc.Texture2DArray.FirstArraySlice = 0;
	uavDesc.Texture2DArray.MipSlice = 0;

	//ComPtr<ID3D11Texture2D> spectrumText;
	EXIT_ON_FAILURE(device->CreateTexture2D(&textDesc, nullptr, &spectrumText));
	EXIT_ON_FAILURE(device->CreateUnorderedAccessView(spectrumText.Get(), &uavDesc, &newSpectrumUAV));
	
	return S_OK;
}

HRESULT OceanClass::CreateScreenMesh(ID3D11Device1 *& device)
{
	float vmargin = 0.1f;
	float hmargin = 0.1f;

	vector<XMFLOAT2> vertices(int(ceil(screenHeight * (1 + 2 * hmargin) / screenGridSize) + 1) * int(ceil(screenWidth * (1 + 2 * vmargin) / screenGridSize) + 1));

	int n = 0;
	int nx;
	for (float j = screenHeight * (1 + hmargin); j > -screenHeight * hmargin - screenGridSize; j -= screenGridSize)
	{
		nx = 0;
		for (float i = -screenWidth * vmargin; i < screenWidth * (1.0 + vmargin) + screenGridSize; i += screenGridSize, ++nx)
		{
			vertices[n++] = XMFLOAT2(-1.0 + 2.0 * i / (float)screenWidth, -1.0 + 2.0 * j / (float)screenHeight);
		}
	}

	stride = sizeof(vertices[0]);
	offset = 0;

	D3D11_BUFFER_DESC vbd;
	vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vbd.ByteWidth = n * sizeof(vertices[0]);
	vbd.CPUAccessFlags = 0;
	vbd.MiscFlags = 0;
	vbd.StructureByteStride = 0;
	vbd.Usage = D3D11_USAGE_IMMUTABLE;

	D3D11_SUBRESOURCE_DATA vinitData;
	vinitData.pSysMem = &vertices[0];

	EXIT_ON_FAILURE(device->CreateBuffer(&vbd, &vinitData, &screenMeshVB));

	// indices
	vector<UINT> indices(6 * int(ceil(screenHeight * (1.0 + 2.0 * hmargin) / screenGridSize) + 1) * int(ceil(screenWidth * (1.0f + 2.0f * hmargin) / screenGridSize) + 1));

	int nj = 0;
	n = 0;
	for (float j = screenHeight * (1.0 + hmargin); j > -screenHeight * hmargin; j -= screenGridSize, ++nj)
	{
		int ni = 0;
		for (float i = -screenWidth * vmargin; i < screenWidth * (1.0 + vmargin); i += screenGridSize, ++ni)
		{
			indices[n++] = ni + (nj + 1) * nx;
			indices[n++] = (ni + 1) + (nj + 1) * nx;
			indices[n++] = (ni + 1) + nj * nx;
			indices[n++] = (ni + 1) + nj * nx;
			indices[n++] = ni + (nj + 1) * nx;
			indices[n++] = ni + nj * nx;
		}
	}

	indicesPerRow = (nx - 1) * 6;

	D3D11_BUFFER_DESC ibd;
	ibd.BindFlags = D3D11_BIND_INDEX_BUFFER;
	ibd.ByteWidth = n * sizeof(indices[0]);
	ibd.CPUAccessFlags = 0;
	ibd.MiscFlags = 0;
	ibd.StructureByteStride = 0;
	ibd.Usage = D3D11_USAGE_IMMUTABLE;

	D3D11_SUBRESOURCE_DATA iinitData;
	iinitData.pSysMem = &indices[0];

	EXIT_ON_FAILURE(device->CreateBuffer(&ibd, &iinitData, &screenMeshIB));

	return S_OK;
}

HRESULT OceanClass::CreateDiscMesh(ID3D11Device1 *& device)
{
	// NOTE: Easier grid resolution setting
	// TODO: Check is drawing from inside out is faster (z-culling etc.)
	int slices = 12;
	float dif = 1.0 / (1 - XM_2PI / slices);
	int steps = 11;

	vector<XMFLOAT2> vertices(steps * slices + 1);
	vector<UINT16> indices(3 * slices*(2 * (steps - 1) + 1));
	vertices[0] = XMFLOAT2(0.0, 0.0);

	int n = 0;
	int in = 0;
	float r = 1.0;
	for (int i = 0; i < steps - 1; ++i, r /= dif)
	{
		for (int j = 0; j < slices; ++j)
		{
			vertices[n++] = XMFLOAT2(r*sin(XM_2PI * j / float(slices)), r*cos(XM_2PI * j / float(slices)));

			indices[in++] = i * slices + j%slices;
			indices[in++] = i * slices + (j + 1) % slices;
			indices[in++] = (i + 1) * slices + (j + 1) % slices;
			indices[in++] = (i + 1) * slices + (j + 1) % slices;
			indices[in++] = (i + 1) * slices + j%slices;
			indices[in++] = i * slices + j%slices;
		}
	}

	for (int j = 0; j < slices; ++j)
	{
		vertices[n++] = XMFLOAT2(r*sin(XM_2PI * j / float(slices)), r*cos(XM_2PI * j / float(slices)));

		indices[in++] = (steps - 1) * slices + j;
		indices[in++] = (steps - 1) * slices + (j + 1) % 12;
		indices[in++] = steps * slices;
	}

	vertices[n] = XMFLOAT2(0.0, 0.0);

	indicesToRender = in;

	D3D11_BUFFER_DESC vbd;
	vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vbd.ByteWidth = vertices.size() * sizeof(vertices[0]);
	vbd.CPUAccessFlags = 0;
	vbd.MiscFlags = 0;
	vbd.StructureByteStride = 0;
	vbd.Usage = D3D11_USAGE_IMMUTABLE;

	D3D11_SUBRESOURCE_DATA vinitData;
	vinitData.pSysMem = &vertices[0];

	EXIT_ON_FAILURE(device->CreateBuffer(&vbd, &vinitData, &discMeshVB));

	D3D11_BUFFER_DESC ibd;
	ibd.BindFlags = D3D11_BIND_INDEX_BUFFER;
	ibd.ByteWidth = indices.size() * sizeof(indices[0]);
	ibd.CPUAccessFlags = 0;
	ibd.MiscFlags = 0;
	ibd.StructureByteStride = 0;
	ibd.Usage = D3D11_USAGE_IMMUTABLE;

	D3D11_SUBRESOURCE_DATA iinitData;
	iinitData.pSysMem = &indices[0];

	EXIT_ON_FAILURE(device->CreateBuffer(&ibd, &iinitData, &discMeshIB));

	return S_OK;
}

HRESULT OceanClass::CreateGridMesh(ID3D11Device1 *& device)
{
	float vmargin = 0.1;
	float hmargin = 0.1;
	float step = 0.1f;

	vector<XMFLOAT2> vertices(101 * 101);

	int n = 0;
	int nx;
	for (float j = -5.0f; j <= 5.0f; j += step)
	{
		nx = 0;
		for (float i = -5.0f; i <= 5.0f; i += step, ++nx)
		{
			vertices[n++] = XMFLOAT2(i, j);
		}
	}

	stride = sizeof(vertices[0]);
	offset = 0;

	D3D11_BUFFER_DESC vbd;
	vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vbd.ByteWidth = n * sizeof(vertices[0]);
	vbd.CPUAccessFlags = 0;
	vbd.MiscFlags = 0;
	vbd.StructureByteStride = 0;
	vbd.Usage = D3D11_USAGE_IMMUTABLE;

	D3D11_SUBRESOURCE_DATA vinitData;
	vinitData.pSysMem = &vertices[0];

	EXIT_ON_FAILURE(device->CreateBuffer(&vbd, &vinitData, &gridMeshVB));

	// indices
	vector<UINT> indices(6 * 100 * 100);

	int nj = 0;
	n = 0;
	for (float j = -5.0f; j < 5.0f; j += step, ++nj)
	{
		int ni = 0;
		for (float i = -5.0f; i < 5.0f; i += step, ++ni)
		{
			indices[n++] = ni + (nj + 1) * nx;
			indices[n++] = (ni + 1) + (nj + 1) * nx;
			indices[n++] = (ni + 1) + nj * nx;
			indices[n++] = (ni + 1) + nj * nx;
			indices[n++] = ni + (nj + 1) * nx;
			indices[n++] = ni + nj * nx;
		}
	}

	//indicesToRender = n;

	D3D11_BUFFER_DESC ibd;
	ibd.BindFlags = D3D11_BIND_INDEX_BUFFER;
	ibd.ByteWidth = n * sizeof(indices[0]);
	ibd.CPUAccessFlags = 0;
	ibd.MiscFlags = 0;
	ibd.StructureByteStride = 0;
	ibd.Usage = D3D11_USAGE_IMMUTABLE;

	D3D11_SUBRESOURCE_DATA iinitData;
	iinitData.pSysMem = &indices[0];

	EXIT_ON_FAILURE(device->CreateBuffer(&ibd, &iinitData, &gridMeshIB));

	return S_OK;
}

HRESULT OceanClass::CreateSamplerRasterDepthStencilStates(ID3D11Device1 *& device)
{
	D3D11_SAMPLER_DESC samplerDesc;
	samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
	samplerDesc.MaxLOD = FLT_MAX;
	samplerDesc.MinLOD = -FLT_MAX;
	samplerDesc.MipLODBias = 0;
	samplerDesc.Filter = D3D11_FILTER_ANISOTROPIC;
	samplerDesc.MaxAnisotropy = 16;
	samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;

	samplerDesc.Filter = D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT;

	samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.Filter = D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT;

	EXIT_ON_FAILURE(device->CreateSamplerState(&samplerDesc, &mSamplerClamp));

	D3D11_RASTERIZER_DESC rastDesc;
	ZeroMemory(&rastDesc, sizeof(D3D11_RASTERIZER_DESC));
	rastDesc.FillMode = D3D11_FILL_SOLID;
	rastDesc.CullMode = D3D11_CULL_NONE;
	rastDesc.FrontCounterClockwise = false;
	rastDesc.DepthClipEnable = false;
	if (FAILED(device->CreateRasterizerState(&rastDesc, &mRastStateSolid))) return false;

	rastDesc.FillMode = D3D11_FILL_WIREFRAME;
	if (FAILED(device->CreateRasterizerState(&rastDesc, &mRastStateFrame))) return false;

	return S_OK;
}

void OceanClass::getSpectrumSample(int i, int j, float lengthScale, float kMin, float * result)
{
	float dk = XM_2PI / lengthScale;
	float kx = i * dk;
	float ky = j * dk;
	if (abs(kx) < kMin && abs(ky) < kMin)
	{
		result[0] = 0.0f;
		result[1] = 0.0f;
	}
	else
	{
		float S = spectrum(kx, ky);
		//float S = PiersonMoskowitzSpectrum(dk);
		//float S = PhillipsSpectrum(kx, ky);
		float h = sqrt(S / 2.0f) * dk;
		float phi = rand01(mt) * XM_2PI;
		result[0] = h * cos(phi);
		result[1] = h * sin(phi);

		result[0] = h;
	}
}

float OceanClass::getSlopeVariance(float kx, float ky, float * spectrum)
{
	float kSquare = kx * kx + ky * ky;
	float real = spectrum[0];
	float img = spectrum[1];
	float hSquare = real * real + img * img;
	return kSquare * hSquare * 2.0f;
}

/*
* Based on work:
* "A unified directional spectrum for long and short wind-driven waves"
* T. Elfouhaily, B. Chapron, K. Katsaros, D. Vandemark
* Journal of Geophysical Research vol 102, p781-796, 1997
*
* with clarification from
* "Real-time Realistic Ocean Lighting using Seamless Transitions from Geometry to BRDF" [FFT version source code]
*/
float OceanClass::spectrum(float kx, float ky, bool omnispectrum)
{
	float U10 = windSpeed;
	float Omega = waveAge;

	// phase speed
	float k = hypot(kx, ky);
	float c = omega(k) / k;

	// peak
	float kp = 9.81f * sqr(Omega / U10);
	float cp = omega(kp) / kp;

	// friction
	float z0 = 3.7e-5f * sqr(U10) / 9.81f * pow(U10 / cp, 0.9f);
	float u_star = 0.41 * U10 / log(10 / z0);

	float alpha_p = 6e-3f * sqrt(Omega);
	float Lpm = exp(-1.25f * sqr(kp / k));
	float gamma = Omega < 1.0f ? 1.7f : 1.7f + 6.0f * log(Omega);
	float sigma = 0.08f * (1.0f + 4.0f * pow(Omega, -3.0f));
	float Gamma = exp(-sqr(sqrt(k / kp) - 1.0f) / (2.0f * sqr(sigma)));
	float Jp = pow(gamma, Gamma);
	float Fp = Lpm * Jp * exp(-Omega / sqrt(10) * (sqrt(k / kp) - 1.0));
	float Bl = alpha_p * (cp / c) * Fp / 2.0f;

	float alpha_m = 0.01f * (u_star < cm ? 1.0f + log(u_star / cm) : 1.0f + 3.0f * log(u_star / cm));
	float Fm = exp(-sqr(k / km - 1) / 4.0f);
	float Bh = 0.5f * alpha_m * cm / c * Fm * Lpm;

	if (omnispectrum)
		return spectrumGain * (Bl + Bh) / (k* sqr(k));

	float a0 = log(2.0f) / 4.0f;
	float ap = 4.0f;
	float am = 0.13f * u_star / cm;
	float delta = tanh(a0 + ap * pow(c / cp, 2.5f) + am * pow(cm / c, 2.5f));

	float phi = atan2(ky, kx);

	if (kx < 0.0f)
		return 0.0f;
	else
	{
		Bl *= 2.0f;
		Bh *= 2.0f;
	}

	return spectrumGain * (Bl + Bh) * (1.0f + delta * cos(2.0f * phi)) / (2.0f * XM_PI * sqr(sqr(k)));
}

float OceanClass::omega(float k)
{
	return sqrt(9.81f * k * (1.0f + pow(k / km, 2)));
}

float OceanClass::PiersonMoskowitzSpectrum(float w)
{
	const float alpha = 0.0081f;
	const float beta = 1.25f;

	return alpha*9.81f*9.81f/pow(w, 5.0f)*exp(-beta*pow(9.81f/(w*windSpeed), 4.0f));
}

float OceanClass::PhillipsSpectrum(float kx, float ky)
{
	float kl = sqrt(kx*kx + ky*ky);
	float dot = kx / kl; // blowing in the <1,0> direction

	float k_length2 = kl  * kl;
	float k_length4 = k_length2 * k_length2;

	float k_dot_w2 = dot * dot;

	float w_length = 20.0;
	float L = w_length * w_length / 9.81;
	float L2 = L * L;

	float damping = 0.001;
	float l2 = L2 * damping * damping;

	return 0.0005 * exp(-1.0f / (k_length2 * L2)) / k_length4 * k_dot_w2;// *exp(-k_length2 * l2);
}

float OceanClass::Fresnel(float dot, float n1, float n2, bool schlick)
{
	if (schlick)
	{
		float R0 = pow((n1 - n2) / (n1 + n2), 2.0);
		return R0 + (1.0 - R0)*pow(1.0 - dot, 5.0);
	}
	else
	{
		float sin_dot = 1.0 - dot*dot;
		float sn2 = sin_dot * pow(n1 / n2, 2.0);
		if (sn2 > 1.0) return 1.0;

		float root = sqrt(1.0 - sin_dot * pow(n1 / n2, 2.0));

		float Rs = pow((n1 * dot - n2 * root) / (n1 * dot + n2 * root), 2.0);
		float Rp = pow((n1 * root - n2 * dot) / (n1 * root + n2 * dot), 2.0);

		return min(max(0.5 * (Rs + Rp), 0.0), 1.0);
	}
}