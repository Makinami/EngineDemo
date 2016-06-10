#include "Ocean.h"

#include <amp.h>

#include "DDSTextureLoader.h"

#include "Utilities\CreateShader.h"
#include "Utilities\CreateBuffer.h"
#include "Utilities\MapResources.h"

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
	GRID_SIZE{ 5488.0, 392.0, 28.0, 2.0 },
	windSpeed(35.0),
	waveAge(0.84),
	cm(0.23),
	km(370.0),
	spectrumGain(2.0),
	time(0),
	rand01(0.0, 1.0),
	screenWidth(1280),
	screenHeight(720),
	screenGridSize(2),
	screen(0)
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

	CreateGridMesh(device);

	CreateScreenMesh(device);

	return S_OK;
}

void OceanClass::Update(ID3D11DeviceContext1 *& mImmediateContext, float dt, DirectionalLight& light, std::shared_ptr<CameraClass> Camera)
{
	time += dt;

	float horizon = max(min(Camera->GetHorizon(), 1.1f), -0.1);
	//horizon = 0.35;
	if (screen)
		indicesToRender = (int)((horizon + 0.1)*screenHeight / screenGridSize) * indicesPerRow;
	
	perFrameParams.dt = dt;
	perFrameParams.time = time;
	perFrameParams.screendy = max(-2.0f*horizon + 2.2f, 0.0);
	XMStoreFloat4x4(&(perFrameParams.screenToCamMatrix), XMMatrixInverse(nullptr, Camera->GetProjTrans()));
	XMStoreFloat4x4(&(perFrameParams.camToWorldMatrix), XMMatrixInverse(nullptr, XMMatrixTranspose(Camera->GetViewMatrix())));
	XMStoreFloat4x4(&(perFrameParams.worldToScreenMatrix), Camera->GetViewProjTransMatrix());
	XMStoreFloat3(&(perFrameParams.camPos), Camera->GetPosition());
	perFrameParams.gridSize = XMFLOAT2(screenGridSize / (float)screenWidth, screenGridSize / (float)screenHeight);
	perFrameParams.sunDir = light.Direction();
	perFrameParams.sunDir.x *= -1; perFrameParams.sunDir.y *= -1; perFrameParams.sunDir.z *= -1;
	perFrameParams.lambdaJ = 4.0f;
	perFrameParams.lambdaV = 2.0;
	perFrameParams.pad = screen;

	XMStoreFloat3(&cameraPos, Camera->GetPosition());
	BuildInstanceBuffer(mImmediateContext, Camera);

	/*instances[0].clear();
	instances[0].push_back(XMFLOAT4X4(
		1.0, 0.0, 0.0, 0.0,
		0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 1.0, 0.0,
		0.0, 0.0, 0.0, 1.0
	));*/

	D3D11_MAPPED_SUBRESOURCE mappedResources;
	mImmediateContext->Map(gridInstancesVB.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResources);
	memcpy(mappedResources.pData, &instances[0][0], sizeof(instances[0][0]) * instances[0].size());
	mImmediateContext->Unmap(gridInstancesVB.Get(), 0);

	ID3D11Buffer* pp = perFrameCB.Get();
	MapResources(mImmediateContext, perFrameCB.Get(), perFrameParams);

	Simulate(mImmediateContext);
}

void OceanClass::Simulate(ID3D11DeviceContext1 *& mImmediateContext)
{
	ID3D11UnorderedAccessView* ppUAViewNULL[2] = { NULL, NULL };
	ID3D11ShaderResourceView* ppSRVNULL[2] = { NULL, NULL };

	ID3D11Buffer* buffers[] = { constCB[2].Get(), perFrameCB.Get() };

	// init
	mImmediateContext->CSSetShader(initFFTCS.Get(), nullptr, 0);
	mImmediateContext->CSSetShaderResources(0, 1, spectrumSRV.GetAddressOf());
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, wavesUAV[0].GetAddressOf(), nullptr);
	mImmediateContext->CSSetConstantBuffers(0, 2, buffers);

	mImmediateContext->Dispatch(FFT_SIZE / 16, FFT_SIZE / 16, 1);

	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);

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

	mImmediateContext->Dispatch(FFT_SIZE / 16, FFT_SIZE / 16, 4);

	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);
	mImmediateContext->CSSetShaderResources(0, 1, ppSRVNULL);

	mImmediateContext->CSSetShader(nullptr, 0, 0);
}

void OceanClass::BuildInstanceBuffer(ID3D11DeviceContext1 *& mImmediateContext, std::shared_ptr<CameraClass> Camera)
{
	float edge = 256000.0;
	instances[0].clear();
	DivideTile(XMFLOAT2(-edge / 2.0, edge / 2.0), edge / 2.0, 0);
	DivideTile(XMFLOAT2(edge / 2.0, edge / 2.0), edge / 2.0, 1);
	DivideTile(XMFLOAT2(-edge / 2.0, -edge / 2.0), edge / 2.0, 2);
	DivideTile(XMFLOAT2(edge / 2.0, -edge / 2.0), edge / 2.0, 3);
}

void OceanClass::DivideTile(XMFLOAT2 pos, float edge, int num)
{
	float dist = sqrt((pos.x - cameraPos.x)*(pos.x - cameraPos.x) + cameraPos.y*cameraPos.y + (pos.y - cameraPos.z)*(pos.y - cameraPos.z));

	if (edge / (dist - edge * sqrt(2.0)) > 10)
	{
		DivideTile(XMFLOAT2(pos.x - edge / 2.0, pos.y + edge / 2.0), edge / 2.0, 0);
		DivideTile(XMFLOAT2(pos.x + edge / 2.0, pos.y + edge / 2.0), edge / 2.0, 1);
		DivideTile(XMFLOAT2(pos.x - edge / 2.0, pos.y - edge / 2.0), edge / 2.0, 2);
		DivideTile(XMFLOAT2(pos.x + edge / 2.0, pos.y - edge / 2.0), edge / 2.0, 3);
	}
	else
	{
		instances[0].push_back(XMFLOAT4X4(
			edge / 100.0, 0.0, 0.0, 0.0,
			0.0, 1.0, 0.0, 0.0,
			0.0, 0.0, edge / 100.0, 0.0,
			-pos.x, 0.0, -pos.y, 1.0
		));
	}
}

void OceanClass::Draw(ID3D11DeviceContext1 *& mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight& light, ID3D11ShaderResourceView* waterB )
{
	ID3D11ShaderResourceView* ppSRVNULL[] = { NULL, NULL, NULL };
	ID3D11Buffer* buffers[] = { constCB[2].Get(), perFrameCB.Get() };

	// IA 
	UINT stride[] = { sizeof(XMFLOAT2), sizeof(XMFLOAT4X4) };
	UINT offset[] = { 0, 0 };
	ID3D11Buffer* vbs[] = { gridMeshVB.Get(), gridInstancesVB.Get() };
	mImmediateContext->IASetInputLayout(mInputLayout.Get());
	mImmediateContext->IASetIndexBuffer((screen ? screenMeshIB : gridMeshIB).Get(), DXGI_FORMAT_R32_UINT, 0);
	mImmediateContext->IASetVertexBuffers(0, 2, vbs, stride, offset);
	mImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	// VS
	mImmediateContext->VSSetShader(mVertexShader.Get(), nullptr, 0);
	mImmediateContext->VSSetConstantBuffers(0, 2, buffers);
	mImmediateContext->VSSetShaderResources(0, 1, wavesSRV[0].GetAddressOf());
	mImmediateContext->VSSetSamplers(1, 1, mSamplerAnisotropic.GetAddressOf());

	// PS 
	mImmediateContext->PSSetShader(mPixelShader.Get(), nullptr, 0);
	mImmediateContext->PSSetConstantBuffers(0, 2, buffers);
	mImmediateContext->PSSetShaderResources(0, 1, wavesSRV[0].GetAddressOf());
	mImmediateContext->PSSetSamplers(1, 1, mSamplerAnisotropic.GetAddressOf());
	mImmediateContext->PSSetShaderResources(1, 1, fresnelSRV.GetAddressOf());
	mImmediateContext->PSSetSamplers(2, 1, mSamplerClamp.GetAddressOf());
	mImmediateContext->PSSetShaderResources(2, 1, turbulenceSRV[turbulenceTextReadId].GetAddressOf());
	mImmediateContext->PSSetShaderResources(3, 1, noiseSRV.GetAddressOf());
	
	// RS & OM
	mImmediateContext->RSSetState(mRastStateSolid.Get());
	mImmediateContext->OMSetDepthStencilState(mDepthStencilState.Get(), 0);

	//mImmediateContext->DrawIndexed(indicesToRender, 0, 0);
	mImmediateContext->DrawIndexedInstanced(indicesToRender, instances[0].size(), 0, 0, 0);

	mImmediateContext->VSSetShaderResources(0, 1, ppSRVNULL);
	mImmediateContext->PSSetShaderResources(0, 3, ppSRVNULL);
}

void OceanClass::Release()
{
}

HRESULT OceanClass::CompileShadersAndInputLayout(ID3D11Device1 *& device)
{
	EXIT_ON_FAILURE(CreateCSFromFile(L"..\\Debug\\Shaders\\Ocean\\initFFT.cso", device, initFFTCS));

	EXIT_ON_FAILURE(CreateCSFromFile(L"..\\Debug\\Shaders\\WaterBruneton\\FFT.cso", device, fftCS));

	EXIT_ON_FAILURE(CreateCSFromFile(L"..\\Debug\\Shaders\\Ocean\\JacobianInject.cso", device, JacobianInjectCS));

	EXIT_ON_FAILURE(CreateCSFromFile(L"..\\Debug\\Shaders\\Ocean\\dissipateTrubulence.cso", device, dissipateTurbulanceCS));
	
	EXIT_ON_FAILURE(CreatePSFromFile(L"..\\Debug\\Shaders\\Ocean\\OceanPS.cso", device, mPixelShader));

	// vertex and input layout
	const D3D11_INPUT_ELEMENT_DESC vertexDesc[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "WORLD", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 0, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
		{ "WORLD", 1, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 16, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
		{ "WORLD", 2, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 32, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
		{ "WORLD", 3, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 48, D3D11_INPUT_PER_INSTANCE_DATA, 1 }
	};

	UINT numElements = sizeof(vertexDesc) / sizeof(vertexDesc[0]);

	EXIT_ON_FAILURE(CreateVSAndInputLayout(L"..\\Debug\\Shaders\\Ocean\\OceanVS.cso", device, mVertexShader, vertexDesc, numElements, mInputLayout));

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
	initData.SysMemPitch = sizeof(buffer);

	// fft 0/row pass
	constCB.push_back(nullptr);
	EXIT_ON_FAILURE(device->CreateBuffer(&constantBufferDesc, &initData, &(constCB.back())));

	// fft 1/col pass
	buffer[0] = 1;
	constCB.push_back(nullptr);
	EXIT_ON_FAILURE(device->CreateBuffer(&constantBufferDesc, &initData, &(constCB.back()))); 
	
	// invert grid size
	float bufferf[8] = { XM_2PI / GRID_SIZE[0], XM_2PI / GRID_SIZE[1], XM_2PI / GRID_SIZE[2], XM_2PI / GRID_SIZE[3],
						 GRID_SIZE[0], GRID_SIZE[1], GRID_SIZE[2], GRID_SIZE[3] };
	constantBufferDesc.ByteWidth = sizeof(bufferf);
	initData.pSysMem = bufferf;
	initData.SysMemPitch = sizeof(buffer);

	constCB.push_back(nullptr);
	EXIT_ON_FAILURE(device->CreateBuffer(&constantBufferDesc, &initData, &(constCB.back())));

	// per frame params
	EXIT_ON_FAILURE(CreateConstantBuffer(device, sizeof(perFrameParams), perFrameCB));

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

	delete[] spectrum;

	/*
	 * SLOPE VARIANCE
	 */
	float sigma2 = 0.01 * (windSpeed <= 7 ? 0.9 + 1.2*log(windSpeed) : -8.4 + 6.0*log(windSpeed));
	sigma2 = max(sigma2, 2e-5);
	//perFrameParams.sigma2 = XMFLOAT2(10.0 / 9.0 * sigma2, 8.0 / 9.0 * sigma2);
	perFrameParams.sigma2 = XMFLOAT2(4.0 / 256.0, 4.0 / 256.0);

	/*
	 * FFT TEXTURES
	 */
	textDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	textDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
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
		fresnel[i] = Fresnel(i / (float)(fresnelRes - 1), 1.0, 1.333);
		fresnel[i + fresnelRes] = Fresnel(i / (float)(fresnelRes - 1), 1.333, 1.0);
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

	/*
	 * INSTANCE BUFFER
	 */
	D3D11_BUFFER_DESC instanceDesc;
	instanceDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	instanceDesc.ByteWidth = sizeof(XMFLOAT4X4) * 200;
	instanceDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	instanceDesc.MiscFlags = 0;
	instanceDesc.StructureByteStride = 0;
	instanceDesc.Usage = D3D11_USAGE_DYNAMIC;

	EXIT_ON_FAILURE(device->CreateBuffer(&instanceDesc, nullptr, &gridInstancesVB));

	/*
	 * NOISE
	 */
	CreateDDSTextureFromFile(device, L"Textures/noise.dds", NULL, &noiseSRV);

	return S_OK;
}

HRESULT OceanClass::CreateScreenMesh(ID3D11Device1 *& device)
{
	float vmargin = 0.1;
	float hmargin = 0.1;

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

HRESULT OceanClass::CreateGridMesh(ID3D11Device1 *& device)
{
	float vmargin = 0.1;
	float hmargin = 0.1;
	float step = 5.0f;

	vector<XMFLOAT2> vertices(401*401);

	int n = 0;
	int nx;
	for (float j = -100.0f; j <= 100.0f; j += step)
	{
		nx = 0;
		for (float i = -100.0f; i <= 100.0f; i += step, ++nx)
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
	vector<UINT> indices(6 * 400*400);

	int nj = 0;
	n = 0;
	for (float j = -100.0f; j < 100.0f; j += step, ++nj)
	{
		int ni = 0;
		for (float i = -100.0f; i < 100.0f; i += step, ++ni)
		{
			indices[n++] = ni + (nj + 1) * nx;
			indices[n++] = (ni + 1) + (nj + 1) * nx;
			indices[n++] = (ni + 1) + nj * nx;
			indices[n++] = (ni + 1) + nj * nx;
			indices[n++] = ni + (nj + 1) * nx;
			indices[n++] = ni + nj * nx;
		}
	}

	indicesToRender = n;

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

	EXIT_ON_FAILURE(device->CreateSamplerState(&samplerDesc, &mSamplerAnisotropic));

	samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.Filter = D3D11_FILTER_MIN_MAG_POINT_MIP_LINEAR;

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
		float h = sqrt(S / 2.0) * dk;
		float phi = rand01(mt) * XM_2PI;
		result[0] = h * cos(phi);
		result[1] = h * sin(phi);
	}
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
	float kp = 9.81 * sqr(Omega / U10);
	float cp = omega(kp) / kp;

	// friction
	float z0 = 3.7e-5 * sqr(U10) / 9.81 * pow(U10 / cp, 0.9);
	float u_star = 0.41 * U10 / log(10 / z0);

	float alpha_p = 6e-3 * sqrt(Omega);
	float Lpm = exp(-1.25 * sqr(kp / k));
	float gamma = Omega < 1.0 ? 1.7 : 1.7 + 6.0 * log(Omega);
	float sigma = 0.08 * (1.0 + 4.0 * pow(Omega, -3.0));
	float Gamma = exp(-sqr(sqrt(k / kp) - 1.0) / (2.0 * sqr(sigma)));
	float Jp = pow(gamma, Gamma);
	float Fp = Lpm * Jp * exp(-Omega / sqrt(10) * (sqrt(k / kp) - 1.0));
	float Bl = alpha_p * (cp / c) * Fp / 2.0;

	float alpha_m = 0.01 * (u_star < cm ? 1.0 + log(u_star / cm) : 1.0 + 3.0 * log(u_star / cm));
	float Fm = exp(-sqr(k / km - 1) / 4.0);
	float Bh = 0.5 * alpha_m * cm / c * Fm * Lpm;

	float a0 = log(2.0) / 4.0;
	float ap = 4.0;
	float am = 0.13 * u_star / cm;
	float delta = tanh(a0 + ap * pow(c / cp, 2.5) + am * pow(cm / c, 2.5));

	float phi = atan2(ky, kx);

	if (kx < 0.0)
		return 0.0;
	else
	{
		Bl *= 2.0;
		Bh *= 2.0;
	}

	return spectrumGain * (Bl + Bh) * (1.0f + delta * cos(2.0f * phi)) / (2.0f * XM_PI * sqr(sqr(k)));

	//float U10 = windSpeed;
	//float Omega = waveAge;

	//// phase speed
	//float k = sqrt(kx * kx + ky * ky);
	//float c = omega(k) / k;

	//// spectral peak
	//float kp = 9.81 * sqr(Omega / U10); // after Eq 3
	//float cp = omega(kp) / kp;

	//// friction velocity
	//float z0 = 3.7e-5 * sqr(U10) / 9.81 * pow(U10 / cp, 0.9f); // Eq 66
	//float u_star = 0.41 * U10 / log(10.0 / z0); // Eq 60

	//float Lpm = exp(-5.0 / 4.0 * sqr(kp / k)); // after Eq 3
	//float gamma = Omega < 1.0 ? 1.7 : 1.7 + 6.0 * log(Omega); // after Eq 3 // log10 or log??
	//float sigma = 0.08 * (1.0 + 4.0 / pow(Omega, 3.0f)); // after Eq 3
	//float Gamma = exp(-1.0 / (2.0 * sqr(sigma)) * sqr(sqrt(k / kp) - 1.0));
	//float Jp = pow(gamma, Gamma); // Eq 3
	//float Fp = Lpm * Jp * exp(-Omega / sqrt(10.0) * (sqrt(k / kp) - 1.0)); // Eq 32
	//float alphap = 0.006 * sqrt(Omega); // Eq 34
	//float Bl = 0.5 * alphap * cp / c * Fp; // Eq 31

	//float alpham = 0.01 * (u_star < cm ? 1.0 + log(u_star / cm) : 1.0 + 3.0 * log(u_star / cm)); // Eq 44
	//float Fm = exp(-0.25 * sqr(k / km - 1.0)); // Eq 41
	//float Bh = 0.5 * alpham * cm / c * Fm * Lpm; // Eq 40 (fixed)

	//if (omnispectrum) {
	//	return (Bl + Bh) / (k * sqr(k)); // Eq 30
	//}

	//float a0 = log(2.0) / 4.0; float ap = 4.0; float am = 0.13 * u_star / cm; // Eq 59
	//float Delta = tanh(a0 + ap * pow(c / cp, 2.5f) + am * pow(cm / c, 2.5f)); // Eq 57

	//float phi = atan2(ky, kx);

	//if (kx < 0.0) {
	//	return 0.0;
	//}
	//else {
	//	Bl *= 2.0;
	//	Bh *= 2.0;
	//}

	//return (Bl + Bh) * (1.0 + Delta * cos(2.0 * phi)) / (2.0 * XM_PI * sqr(sqr(k))); // Eq 67
}

float OceanClass::omega(float k)
{
	return sqrt(9.81 * k * (1.0 + pow(k / km, 2)));
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
		float root = sqrt(1.0 - pow(sin_dot * n1 / n2, 2.0));

		float Rs = pow((n1 * dot - n2 * root) / (n1 * dot + n2 * root), 2.0);
		float Rp = pow((n1 * root - n2 * dot) / (n1 * root + n2 * dot), 2.0);

		return clamp(0.5 * (Rs + Rp), 0.0, 1.0);
	}	
}