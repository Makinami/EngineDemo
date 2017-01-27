#include "WaterBruneton.h"

#include "Utilities\CreateShader.h"
#include "Utilities\CreateBuffer.h"
#include "Utilities\MapResources.h"

#include "RenderStates.h"

const float M_PI = 3.14159265;

WaterBruneton::WaterBruneton() :
	FFT_SIZE(256),
	GRID_SIZE{ 5488.0, 392.0, 28.0, 2.0 },
	N_SLOPE_VARIANCE(16),
	mt(rd()),
	rand01(),
	cm(0.23),
	km(370.0),
	WIND(5.0), // wind speed in m/s (at 10m above surface)
	OMEGA(0.84), // sea state (inverse wave age)
	A(1.0), // wave aplitude factor (should be one?)
	time(0),
	spectrumSRV(0),
	slopeVarianceSRV(0),
	slopeVarianceUAV(0),
	fftWavesSRV(0),
	fftWavesSRVTemp(0),
	fftWavesUAV(0),
	fftWavesUAVTemp(0),
	variancesCS(0),
	initFFTCS(0),
	fftCS(0),
	variancesCB(0),
	initFFTCB(0),
	fftCB(0),
	mScreenMeshIB(0),
	mScreenMeshVB(0),
	mScreenHeight(720),
	mScreenWidth(1280),
	horizon(0.5),
	screenGridSize(8),
	mNumScreenMeshIndices(0)
{
}


WaterBruneton::~WaterBruneton()
{
	ReleaseCOM(spectrumSRV);
	ReleaseCOM(slopeVarianceSRV);
	ReleaseCOM(slopeVarianceUAV);
	ReleaseCOM(fftWavesSRV);
	ReleaseCOM(fftWavesSRVTemp);
	ReleaseCOM(fftWavesUAV);
	ReleaseCOM(fftWavesUAVTemp);
	ReleaseCOM(variancesCS);
	ReleaseCOM(initFFTCS);
	ReleaseCOM(fftCS);
	ReleaseCOM(BinitFFTCS);
	ReleaseCOM(Bfftx);
	ReleaseCOM(Bffty);
	ReleaseCOM(spectrum12SRV);
	ReleaseCOM(spectrum34SRV);
	ReleaseCOM(BbutterflySRV);
	ReleaseCOM(mInputLayout);
	ReleaseCOM(mVertexShader);
	ReleaseCOM(mPixelShader);
	ReleaseCOM(variancesCB);
	ReleaseCOM(initFFTCB);
	ReleaseCOM(fftCB);
	ReleaseCOM(drawCB);
	ReleaseCOM(mScreenMeshIB);
	ReleaseCOM(mScreenMeshVB);

	ReleaseCOM(mRastStateFrame);
	ReleaseCOM(mSamplerState);
	ReleaseCOM(mSamplerAnisotropic);
	ReleaseCOM(mSSSlopeVariance);
	ReleaseCOM(mDepthStencilStateSea);
}

bool WaterBruneton::Init(ID3D11Device1 * &device, ID3D11DeviceContext1 * &mImmediateContext)
{
	if (!CreateDataResources(device)) return false;

	if (!CreateInputLayoutAndShaders(device)) return false;

	//BCreatespectrumEtc(device);

	ComputeVarianceText(mImmediateContext);

	mDevice = device;
	GenerateScreenMesh();

	D3D11_RASTERIZER_DESC rastDesc;
	ZeroMemory(&rastDesc, sizeof(D3D11_RASTERIZER_DESC));
	rastDesc.FillMode = D3D11_FILL_SOLID;
	rastDesc.CullMode = D3D11_CULL_NONE;
	rastDesc.FrontCounterClockwise = false;
	rastDesc.DepthClipEnable = false;

	if (FAILED(device->CreateRasterizerState(&rastDesc, &mRastStateFrame))) return false;

	computeFFTPrf = Performance->ReserveName(L"Bruneton FFT");
	drawPrf = Performance->ReserveName(L"Bruneton Draw");

	// depth stencil state
	D3D11_DEPTH_STENCIL_DESC dsDesc;
	dsDesc.DepthEnable = true;
	dsDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	dsDesc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
	dsDesc.StencilEnable = false;
	
	device->CreateDepthStencilState(&dsDesc, &mDepthStencilStateSea);
	
	return true;
}

void WaterBruneton::Draw(ID3D11DeviceContext1 * &mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight& light)
{
	float _horizon = Camera->GetHorizon();
	if (horizon != _horizon)
	{
		horizon = _horizon;
		GenerateScreenMesh();
	}

	CallStart(drawPrf);

	ID3D11ShaderResourceView* ppSRVNULL[2] = { NULL, NULL };

	UINT stride = sizeof(XMFLOAT3);
	UINT offset = 0;
	
	XMStoreFloat4x4(&(drawParams.screenToCamera), XMMatrixInverse(nullptr, Camera->GetProjTrans()));
	XMStoreFloat4x4(&(drawParams.cameraToWorld), XMMatrixInverse(nullptr, XMMatrixTranspose(Camera->GetViewMatrix())));
	XMStoreFloat4x4(&(drawParams.worldToScreen), Camera->GetViewProjTransMatrix());
	XMStoreFloat3(&(drawParams.worldCamera), Camera->GetPosition());
	drawParams.GRID_SIZE = XMFLOAT4(GRID_SIZE);
	drawParams.choppy = 1.0;
	drawParams.gridSize = XMFLOAT2(screenGridSize / (float)mScreenWidth, screenGridSize / (float)mScreenHeight);
	drawParams.worldSunDir = light.Direction();
	// change light direction to sun direction
	drawParams.worldSunDir.x *= -1; drawParams.worldSunDir.y *= -1; drawParams.worldSunDir.z *= -1;
	drawParams.seaColour = XMFLOAT3(1.0f / 255.0f, 4.0f / 255.0f, 12.0f / 255.0f);

	MapResources(mImmediateContext, drawCB, drawParams);

	// PS
	mImmediateContext->PSSetShader(mPixelShader, nullptr, 0);
	mImmediateContext->PSSetConstantBuffers(0, 1, &drawCB);
	mImmediateContext->PSSetShaderResources(0, 1, &fftWavesSRV);
	mImmediateContext->PSSetShaderResources(1, 1, &slopeVarianceSRV);
	mImmediateContext->PSSetSamplers(2, 1, &RenderStates::Sampler::AnisotropicWrapSS);
	mImmediateContext->PSSetSamplers(3, 1, &RenderStates::Sampler::TrilinearClampSS);

	// VS
	mImmediateContext->VSSetShader(mVertexShader, nullptr, 0);
	mImmediateContext->VSSetConstantBuffers(0, 1, &drawCB);
	mImmediateContext->VSSetShaderResources(0, 1, &fftWavesSRV);
	mImmediateContext->VSSetSamplers(2, 1, &RenderStates::Sampler::AnisotropicWrapSS);

	// IA
	mImmediateContext->IASetInputLayout(mInputLayout);

	mImmediateContext->IASetIndexBuffer(mScreenMeshIB, DXGI_FORMAT_R32_UINT, 0);
	mImmediateContext->IASetVertexBuffers(0, 1, &mScreenMeshVB, &stride, &offset);
	mImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	mImmediateContext->RSSetState(mRastStateFrame);
	mImmediateContext->OMSetDepthStencilState(mDepthStencilStateSea, 0);

	mImmediateContext->DrawIndexed(mNumScreenMeshIndices, 0, 0);

	mImmediateContext->VSSetShaderResources(0, 2, ppSRVNULL);
	mImmediateContext->PSSetShaderResources(0, 2, ppSRVNULL);

	CallEnd(drawPrf);
}

void WaterBruneton::EvaluateWaves(float t, ID3D11DeviceContext1 * &mImmediateContext)
{
	//ComputeVarianceText(mImmediateContext);

	Performance->Call(computeFFTPrf, Debug::PerformanceClass::CallType::START);

	time += t;

	ID3D11UnorderedAccessView* ppUAViewNULL[2] = { NULL, NULL };
	ID3D11ShaderResourceView* ppSRVNULL[2] = { NULL, NULL };

	// init
	mImmediateContext->CSSetShader(initFFTCS, nullptr, 0);
	mImmediateContext->CSSetShaderResources(0, 1, &spectrumSRV);
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, &fftWavesUAV, nullptr);

	initFFTParams.INVERSE_GRID_SIZE = XMFLOAT4(XM_2PI / GRID_SIZE[0], XM_2PI / GRID_SIZE[1], XM_2PI / GRID_SIZE[2], XM_2PI / GRID_SIZE[3]);
	initFFTParams.time = time;
	MapResources(mImmediateContext, initFFTCB, initFFTParams);
	mImmediateContext->CSSetConstantBuffers(0, 1, &initFFTCB);

	mImmediateContext->Dispatch(ceilf(FFT_SIZE / 16.0f), ceilf(FFT_SIZE / 16.0f), 1);

	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);
	mImmediateContext->CSSetShaderResources(0, 1, ppSRVNULL);

	// 1st pass
	mImmediateContext->CSSetShader(fftCS, nullptr, 0);
	mImmediateContext->CSSetShaderResources(0, 1, &fftWavesSRV);
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, &fftWavesUAVTemp, nullptr);

	fftParams.col_row_pass = fftParams.ROW_PASS;
	MapResources(mImmediateContext, fftCB, fftParams);
	mImmediateContext->CSSetConstantBuffers(0, 1, &fftCB);

	mImmediateContext->Dispatch(1, 256, 6);

	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);

	//2nd pass
	mImmediateContext->CSSetShaderResources(0, 1, &fftWavesSRVTemp);
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, &fftWavesUAV, nullptr);

	fftParams.col_row_pass = fftParams.COL_PASS;
	MapResources(mImmediateContext, fftCB, fftParams);
	mImmediateContext->CSSetConstantBuffers(0, 1, &fftCB);

	mImmediateContext->Dispatch(1, 256, 6);

	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);
	mImmediateContext->CSSetShaderResources(0, 1, ppSRVNULL);

	Performance->Call(computeFFTPrf, Debug::PerformanceClass::CallType::END);
}

void WaterBruneton::BEvelWater(float t, ID3D11DeviceContext1 * mImmediateContext)
{
	Performance->Call(computeFFTPrf, Debug::PerformanceClass::CallType::START);

	time += t;

	ID3D11UnorderedAccessView* ppUAViewNULL[2] = { NULL, NULL };
	ID3D11ShaderResourceView* ppSRVNULL[2] = { NULL, NULL };

	// init
	mImmediateContext->CSSetShader(BinitFFTCS, nullptr, 0);
	mImmediateContext->CSSetShaderResources(0, 1, &spectrum12SRV);
	mImmediateContext->CSSetShaderResources(1, 1, &spectrum34SRV);
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, &fftWavesUAV, nullptr);

	initFFTParams.INVERSE_GRID_SIZE = XMFLOAT4(1.0f / GRID_SIZE[0], 1.0f / GRID_SIZE[1], 1.0f / GRID_SIZE[2], 1.0f / GRID_SIZE[3]);
	initFFTParams.time = time;
	MapResources(mImmediateContext, initFFTCB, initFFTParams);
	mImmediateContext->CSSetConstantBuffers(0, 1, &initFFTCB);

	mImmediateContext->Dispatch(ceilf(FFT_SIZE / 16.0f), ceilf(FFT_SIZE / 16.0f), 1);

	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);
	mImmediateContext->CSSetShaderResources(0, 1, ppSRVNULL);

	// 1st pass
	mImmediateContext->CSSetShader(Bfftx, nullptr, 0);
	mImmediateContext->CSSetShaderResources(1, 1, &BbutterflySRV);
	int i;
	for (i = 0; i < 8; ++i)
	{
		fftParams.col_row_pass = i;
		MapResources(mImmediateContext, fftCB, fftParams);
		if (i % 2 == 0)
		{
			mImmediateContext->CSSetShaderResources(0, 1, &fftWavesSRV);
			mImmediateContext->CSSetUnorderedAccessViews(0, 1, &fftWavesUAVTemp, nullptr);
		}
		else
		{
			mImmediateContext->CSSetShaderResources(0, 1, &fftWavesSRVTemp);
			mImmediateContext->CSSetUnorderedAccessViews(0, 1, &fftWavesUAV, nullptr);
		}

		mImmediateContext->Dispatch(1, 256, 1);

		mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);
	}

	//2nd pass
	mImmediateContext->CSSetShader(Bffty, nullptr, 0);
	for (i = 8; i < 16; ++i)
	{
		fftParams.col_row_pass = i - 8;
		MapResources(mImmediateContext, fftCB, fftParams);
		if (i % 2 == 0)
		{
			mImmediateContext->CSSetShaderResources(0, 1, &fftWavesSRV);
			mImmediateContext->CSSetUnorderedAccessViews(0, 1, &fftWavesUAVTemp, nullptr);
		}
		else
		{
			mImmediateContext->CSSetShaderResources(0, 1, &fftWavesSRVTemp);
			mImmediateContext->CSSetUnorderedAccessViews(0, 1, &fftWavesUAV, nullptr);
		}

		mImmediateContext->Dispatch(1, 256, 1);

		mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);
	}

	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);
	mImmediateContext->CSSetShaderResources(0, 2, ppSRVNULL);

	Performance->Call(computeFFTPrf, Debug::PerformanceClass::CallType::END);
}

int bitReverse(int i, int N)
{
	int j = i;
	int M = N;
	int Sum = 0;
	int W = 1;
	M = M / 2;
	while (M != 0) {
		j = (i & M) > M - 1;
		Sum += j * W;
		W *= 2;
		M = M / 2;
	}
	return Sum;
}

void computeWeight(int N, int k, float &Wr, float &Wi)
{
	Wr = cosl(2.0 * M_PI * k / float(N));
	Wi = sinl(2.0 * M_PI * k / float(N));
}

float *computeButterflyLookupTexture()
{
	float *data = new float[256 * 8 * 4];

	for (int i = 0; i < 8; i++) {
		int nBlocks = (int)powf(2.0, float(8 - 1 - i));
		int nHInputs = (int)powf(2.0, float(i));
		for (int j = 0; j < nBlocks; j++) {
			for (int k = 0; k < nHInputs; k++) {
				int i1, i2, j1, j2;
				if (i == 0) {
					i1 = j * nHInputs * 2 + k;
					i2 = j * nHInputs * 2 + nHInputs + k;
					j1 = bitReverse(i1, 256);
					j2 = bitReverse(i2, 256);
				}
				else {
					i1 = j * nHInputs * 2 + k;
					i2 = j * nHInputs * 2 + nHInputs + k;
					j1 = i1;
					j2 = i2;
				}

				float wr, wi;
				computeWeight(256, k * nBlocks, wr, wi);

				int offset1 = 4 * (i1 + i * 256);
				data[offset1 + 0] = (j1 + 0.5) / 256;
				data[offset1 + 1] = (j2 + 0.5) / 256;
				data[offset1 + 2] = wr;
				data[offset1 + 3] = wi;

				int offset2 = 4 * (i2 + i * 256);
				data[offset2 + 0] = (j1 + 0.5) / 256;
				data[offset2 + 1] = (j2 + 0.5) / 256;
				data[offset2 + 2] = -wr;
				data[offset2 + 3] = -wi;
			}
		}
	}

	return data;
}

bool WaterBruneton::BCreatespectrumEtc(ID3D11Device1 * device)
{
	float* spectrum12 = new float[FFT_SIZE * FFT_SIZE * 4];
	float* spectrum34 = new float[FFT_SIZE * FFT_SIZE * 4];

	for (int y = 0; y < FFT_SIZE; ++y) {
		for (int x = 0; x < FFT_SIZE; ++x) {
			int offset = 4 * (x + y * FFT_SIZE);
			int i = x >= FFT_SIZE / 2 ? x - FFT_SIZE : x;
			int j = y >= FFT_SIZE / 2 ? y - FFT_SIZE : y;
			getSpectrumSample(i, j, GRID_SIZE[0], M_PI / GRID_SIZE[0], spectrum12 + offset);
			getSpectrumSample(i, j, GRID_SIZE[1], M_PI * FFT_SIZE / GRID_SIZE[1], spectrum12 + offset + 2);
			getSpectrumSample(i, j, GRID_SIZE[2], M_PI * FFT_SIZE / GRID_SIZE[2], spectrum34 + offset);
			getSpectrumSample(i, j, GRID_SIZE[3], M_PI * FFT_SIZE / GRID_SIZE[3], spectrum34 + offset + 2);
		}
	}

	// 12
	D3D11_TEXTURE2D_DESC spectrumDesc = {};
	spectrumDesc.Width = FFT_SIZE;
	spectrumDesc.Height = FFT_SIZE;
	spectrumDesc.MipLevels = 1;
	spectrumDesc.ArraySize = 1;
	spectrumDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	spectrumDesc.CPUAccessFlags - 0;
	spectrumDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	spectrumDesc.MiscFlags = 0;
	spectrumDesc.SampleDesc = { 1, 0 };
	spectrumDesc.Usage = D3D11_USAGE_IMMUTABLE;

	D3D11_SUBRESOURCE_DATA spectrumData;
	spectrumData.pSysMem = spectrum12;
	spectrumData.SysMemPitch = sizeof(float)* FFT_SIZE * 4;

	ComPtr<ID3D11Texture2D> spectrumTex;
	if (FAILED(device->CreateTexture2D(&spectrumDesc, &spectrumData, &spectrumTex))) return false;

	D3D11_SHADER_RESOURCE_VIEW_DESC spectrumSRVDesc;
	spectrumSRVDesc.Format = spectrumDesc.Format;
	spectrumSRVDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	spectrumSRVDesc.Texture2D.MipLevels = 1;
	spectrumSRVDesc.Texture2D.MostDetailedMip = 0;

	if ((device->CreateShaderResourceView(spectrumTex.Get(), &spectrumSRVDesc, &spectrum12SRV))) return false;
	
	// 34
	spectrumData.pSysMem = spectrum34;

	if (FAILED(device->CreateTexture2D(&spectrumDesc, &spectrumData, &spectrumTex))) return false;
	
	if ((device->CreateShaderResourceView(spectrumTex.Get(), &spectrumSRVDesc, &spectrum34SRV))) return false;

	delete[] spectrum12;
	delete[] spectrum34;

	// butterfly
	spectrumDesc.Height = 8;
	spectrumDesc.ArraySize = 1;

	spectrumData.pSysMem = computeButterflyLookupTexture();

	if (FAILED(device->CreateTexture2D(&spectrumDesc, &spectrumData, &spectrumTex))) return false;

	if ((device->CreateShaderResourceView(spectrumTex.Get(), &spectrumSRVDesc, &BbutterflySRV))) return false;

	CreateCSFromFile(L"..\\Debug\\Shaders\\WaterBruneton\\BinitFFT.cso", device, BinitFFTCS);
	CreateCSFromFile(L"..\\Debug\\Shaders\\WaterBruneton\\Bffty.cso", device, Bffty);
	CreateCSFromFile(L"..\\Debug\\Shaders\\WaterBruneton\\Bfftx.cso", device, Bfftx);

	return true;
}

bool WaterBruneton::CreateDataResources(ID3D11Device1 * device)
{
	/*
	 * SPECTRUM
	 */
	unsigned int SLICE_SIZE = FFT_SIZE * FFT_SIZE * 2;
	float* spectrum = new float[SLICE_SIZE * 4];

	// populate wave spectrum table and calculate total slope variance
	totalSlopeVariance = 0.0f;
	for (int y = 0; y < FFT_SIZE; ++y)
	{
		for (int x = 0; x < FFT_SIZE; ++x)
		{
			int offset = 2 * (y * FFT_SIZE + x);
			// F[n] == F[N-n]
			int i = x >= FFT_SIZE / 2 ? x - FFT_SIZE : x;
			int j = y >= FFT_SIZE / 2 ? y - FFT_SIZE : y;
			// populate spectrum
			getSpectrumSample(i, j, GRID_SIZE[0], XM_PI / GRID_SIZE[0], spectrum + offset);
			getSpectrumSample(i, j, GRID_SIZE[1], XM_PI * FFT_SIZE / GRID_SIZE[0], spectrum + SLICE_SIZE + offset);
			getSpectrumSample(i, j, GRID_SIZE[2], XM_PI * FFT_SIZE / GRID_SIZE[1], spectrum + 2 * SLICE_SIZE + offset);
			getSpectrumSample(i, j, GRID_SIZE[3], XM_PI * FFT_SIZE / GRID_SIZE[2], spectrum + 3 * SLICE_SIZE + offset);
			// sum slope variance
			totalSlopeVariance += getSlopeVariance(i * XM_2PI / GRID_SIZE[0], j * XM_2PI / GRID_SIZE[0], spectrum + offset);
			totalSlopeVariance += getSlopeVariance(i * XM_2PI / GRID_SIZE[1], j * XM_2PI / GRID_SIZE[1], spectrum + SLICE_SIZE + offset);
			totalSlopeVariance += getSlopeVariance(i * XM_2PI / GRID_SIZE[2], j * XM_2PI / GRID_SIZE[2], spectrum + 2 * SLICE_SIZE + offset);
			totalSlopeVariance += getSlopeVariance(i * XM_2PI / GRID_SIZE[3], j * XM_2PI / GRID_SIZE[3], spectrum + 3 * SLICE_SIZE + offset);
		}
	}

	// theoretical slope variance
	theoreticalSlopeVariance = 0.0f;
	float k = 5e-3;
	while (k < 1e3)
	{
		float nextK = k * 1.001;
		theoreticalSlopeVariance += k * k * this->spectrum(k, 0, true) * (nextK - k);
		k = nextK;
	}

	D3D11_TEXTURE2D_DESC spectrumDesc = {};
	spectrumDesc.Width = FFT_SIZE;
	spectrumDesc.Height = FFT_SIZE;
	spectrumDesc.MipLevels = 1;
	spectrumDesc.ArraySize = 4;
	spectrumDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	spectrumDesc.CPUAccessFlags - 0;
	spectrumDesc.Format = DXGI_FORMAT_R32G32_FLOAT;
	spectrumDesc.MiscFlags = 0;
	spectrumDesc.SampleDesc = { 1, 0 };
	spectrumDesc.Usage = D3D11_USAGE_IMMUTABLE;
	
	D3D11_SUBRESOURCE_DATA spectrumData[4];
	for (int i = 0; i < 4; ++i)
	{
		spectrumData[i].pSysMem = spectrum + i * SLICE_SIZE;
		spectrumData[i].SysMemPitch = sizeof(float) * FFT_SIZE * 2;
	}

	ComPtr<ID3D11Texture2D> spectrumTex;
	if (FAILED(device->CreateTexture2D(&spectrumDesc, spectrumData, &spectrumTex))) return false;

	
	
	D3D11_SHADER_RESOURCE_VIEW_DESC spectrumSRVDesc;
	spectrumSRVDesc.Format = spectrumDesc.Format;
	spectrumSRVDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
	spectrumSRVDesc.Texture2DArray.ArraySize = spectrumDesc.ArraySize;
	spectrumSRVDesc.Texture2DArray.FirstArraySlice = 0;
	spectrumSRVDesc.Texture2DArray.MipLevels = spectrumDesc.MipLevels;
	spectrumSRVDesc.Texture2DArray.MostDetailedMip = 0;

	if ((device->CreateShaderResourceView(spectrumTex.Get(), &spectrumSRVDesc, &spectrumSRV))) return false;

	delete[] spectrum;

	/*
	 * SLOPE VARIANCE
	 */
	D3D11_TEXTURE3D_DESC slopeVarianceDesc = {};
	slopeVarianceDesc.Width = N_SLOPE_VARIANCE;
	slopeVarianceDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	slopeVarianceDesc.CPUAccessFlags = 0;
	slopeVarianceDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	slopeVarianceDesc.Height = N_SLOPE_VARIANCE;
	slopeVarianceDesc.MipLevels = 1;
	slopeVarianceDesc.MiscFlags = 0;
	slopeVarianceDesc.Usage = D3D11_USAGE_DEFAULT;
	slopeVarianceDesc.Depth = N_SLOPE_VARIANCE;

	ComPtr<ID3D11Texture3D> slopeVarianceTex;
	if (FAILED(device->CreateTexture3D(&slopeVarianceDesc, nullptr, &slopeVarianceTex))) return false;

	D3D11_SHADER_RESOURCE_VIEW_DESC slopeVarianceSRVDesc;
	slopeVarianceSRVDesc.Format = slopeVarianceDesc.Format;
	slopeVarianceSRVDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE3D;
	slopeVarianceSRVDesc.Texture3D.MipLevels = slopeVarianceDesc.MipLevels;
	slopeVarianceSRVDesc.Texture3D.MostDetailedMip = 0;

	if (FAILED(device->CreateShaderResourceView(slopeVarianceTex.Get(), &slopeVarianceSRVDesc, &slopeVarianceSRV))) return false;

	D3D11_UNORDERED_ACCESS_VIEW_DESC slopeVarianceUAVDesc;
	slopeVarianceUAVDesc.Format = slopeVarianceDesc.Format;
	slopeVarianceUAVDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE3D;
	slopeVarianceUAVDesc.Texture3D.MipSlice = 0;
	slopeVarianceUAVDesc.Texture3D.FirstWSlice = 0;
	slopeVarianceUAVDesc.Texture3D.WSize = slopeVarianceDesc.Depth;

	if (FAILED(device->CreateUnorderedAccessView(slopeVarianceTex.Get(), &slopeVarianceUAVDesc, &slopeVarianceUAV))) return false;

	/*
	 * FFTWAVES & TEMP
	 */
	D3D11_TEXTURE2D_DESC fftWavesDesc = {};
	fftWavesDesc.ArraySize = 6;
	fftWavesDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	fftWavesDesc.CPUAccessFlags = 0;
	fftWavesDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	fftWavesDesc.Height = FFT_SIZE;
	fftWavesDesc.MipLevels = 1;
	fftWavesDesc.MiscFlags = 0;
	fftWavesDesc.SampleDesc = { 1, 0 };
	fftWavesDesc.Usage = D3D11_USAGE_DEFAULT;
	fftWavesDesc.Width = FFT_SIZE;

	D3D11_SHADER_RESOURCE_VIEW_DESC fftWavesSRVDesc;
	fftWavesSRVDesc.Format = fftWavesDesc.Format;
	fftWavesSRVDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
	fftWavesSRVDesc.Texture2DArray.ArraySize = fftWavesDesc.ArraySize;
	fftWavesSRVDesc.Texture2DArray.FirstArraySlice = 0;
	fftWavesSRVDesc.Texture2DArray.MipLevels = fftWavesDesc.MipLevels;
	fftWavesSRVDesc.Texture2DArray.MostDetailedMip = 0;

	D3D11_UNORDERED_ACCESS_VIEW_DESC fftWavesUAVDesc;
	fftWavesUAVDesc.Format = fftWavesDesc.Format;
	fftWavesUAVDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2DARRAY;
	fftWavesUAVDesc.Texture2DArray.ArraySize = 6;
	fftWavesUAVDesc.Texture2DArray.FirstArraySlice = 0;
	fftWavesUAVDesc.Texture2DArray.MipSlice = 0;

	ComPtr<ID3D11Texture2D> fftWavesTex;
	// main fft texture
	if (FAILED(device->CreateTexture2D(&fftWavesDesc, nullptr, &fftWavesTex))) return false;
	if (FAILED(device->CreateShaderResourceView(fftWavesTex.Get(), &fftWavesSRVDesc, &fftWavesSRV))) return false;
	if (FAILED(device->CreateUnorderedAccessView(fftWavesTex.Get(), &fftWavesUAVDesc, &fftWavesUAV))) return false;
	fftWavesTex = nullptr;
	// secondary fft texture
	if (FAILED(device->CreateTexture2D(&fftWavesDesc, nullptr, &fftWavesTex))) return false;
	if (FAILED(device->CreateShaderResourceView(fftWavesTex.Get(), &fftWavesSRVDesc, &fftWavesSRVTemp))) return false;
	if (FAILED(device->CreateUnorderedAccessView(fftWavesTex.Get(), &fftWavesUAVDesc, &fftWavesUAVTemp))) return false;

	/*
	 * CONSTANT BUFFERS
	 */
	// variances
	if (!CreateConstantBuffer(device, sizeof(variancesCBType), variancesCB)) return false;

	// init fft params
	if (!CreateConstantBuffer(device, sizeof(initFFTCBType), initFFTCB)) return false;

	// fft params
	if (!CreateConstantBuffer(device, sizeof(fftCBType), fftCB)) return false;

	// draw params
	if (!CreateConstantBuffer(device, sizeof(vertexShaderCBType), drawCB)) return false;
	
	return true;
}

bool WaterBruneton::CreateInputLayoutAndShaders(ID3D11Device1 * device)
{
	// variances
	if (!CreateCSFromFile(L"..\\Debug\\Shaders\\WaterBruneton\\variances.cso", device, variancesCS)) return false;

	// init FFT
	if (!CreateCSFromFile(L"..\\Debug\\Shaders\\WaterBruneton\\initFFT.cso", device, initFFTCS)) return false;;

	// fft
	if (!CreateCSFromFile(L"..\\Debug\\Shaders\\WaterBruneton\\FFT.cso", device, fftCS)) return false;

	// pixel
	if (!CreatePSFromFile(L"..\\Debug\\Shaders\\WaterBruneton\\WaterPS.cso", device, mPixelShader)) return false;

	// vertex and input layout
	D3D11_INPUT_ELEMENT_DESC vertexDesc[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 }
	};

	int numElements = sizeof(vertexDesc) / sizeof(vertexDesc[0]);

	if (!CreateVSAndInputLayout(L"..\\Debug\\Shaders\\WaterBruneton\\WaterVS.cso", device, mVertexShader, vertexDesc, numElements, mInputLayout)) return false;

	return true;
}

void WaterBruneton::ComputeVarianceText(ID3D11DeviceContext1* mImmediateContext)
{
	ID3D11UnorderedAccessView* ppUAViewNULL[2] = { nullptr, nullptr };
	ID3D11ShaderResourceView* ppSRVNULL[2] = { nullptr, nullptr };

	mImmediateContext->CSSetShader(variancesCS, nullptr, 0);
	mImmediateContext->CSSetShaderResources(0, 1, &spectrumSRV);
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, &slopeVarianceUAV, nullptr);
	mImmediateContext->CSSetSamplers(0, 1, &RenderStates::Sampler::TrilinearWrapSS);

	variancesParams.GRID_SIZE = XMFLOAT4(GRID_SIZE);
	variancesParams.slopeVarianceDelta = 0.5 * (theoreticalSlopeVariance - totalSlopeVariance);
	MapResources(mImmediateContext, variancesCB, variancesParams);
	mImmediateContext->CSSetConstantBuffers(0, 1, &variancesCB);

	mImmediateContext->Dispatch(1, 1, N_SLOPE_VARIANCE);

	mImmediateContext->CSSetUnorderedAccessViews(0, 1, ppUAViewNULL, nullptr);
	mImmediateContext->CSSetShaderResources(0, 1, ppSRVNULL);
}

void WaterBruneton::GenerateScreenMesh()
{
	ReleaseCOM(mScreenMeshIB);
	ReleaseCOM(mScreenMeshVB);
	mNumScreenMeshIndices = 0;

	if (horizon <= 0.0f) return;

	float s = min(horizon, 1.1f);

	float vmargin = 0.1;
	float hmargin = 0.1;

	// vertices
	XMFLOAT3* vertices = new XMFLOAT3[int(ceil(mScreenHeight * (s + hmargin) / screenGridSize) + 2) * int(ceil(mScreenWidth * (1.0f + 2.0f * hmargin) / screenGridSize) + 2)];

	int n = 0;
	int nx = 0;
	for (float j = mScreenHeight * s; j > -mScreenHeight * hmargin - screenGridSize; j -= screenGridSize)
	{
		nx = 0;
		for (float i = -mScreenWidth * hmargin; i < mScreenWidth * (1.0 + vmargin) + screenGridSize; i += screenGridSize, ++nx)
		{
			vertices[n++] = XMFLOAT3(-1.0 + 2.0 * i / (float)mScreenWidth, -1.0 + 2.0 * j / (float)mScreenHeight, 0.0);
		}
	}

	D3D11_BUFFER_DESC vbd;
	vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vbd.ByteWidth = n * sizeof(XMFLOAT3);
	vbd.CPUAccessFlags = 0;
	vbd.MiscFlags = 0;
	vbd.StructureByteStride = 0;
	vbd.Usage = D3D11_USAGE_IMMUTABLE;
	
	D3D11_SUBRESOURCE_DATA vinitData;
	vinitData.pSysMem = vertices;

	mDevice->CreateBuffer(&vbd, &vinitData, &mScreenMeshVB);

	delete[] vertices;

	// indices
	UINT* indices = new UINT[6 * int(ceil(mScreenHeight * (s + hmargin) / screenGridSize) + 1) * int(ceil(mScreenWidth * (1.0f + 2.0f * hmargin) / screenGridSize) + 1)];

	int nj = 0;
	for (float j = mScreenHeight * s; j > -mScreenHeight * hmargin; j -= screenGridSize, ++nj)
	{
		int ni = 0;
		for (float i = -mScreenWidth * vmargin; i < mScreenWidth * (1.0 + vmargin); i += screenGridSize, ++ni)
		{
			indices[mNumScreenMeshIndices++] = ni + (nj + 1) * nx;
			indices[mNumScreenMeshIndices++] = (ni + 1) + (nj + 1) * nx;
			indices[mNumScreenMeshIndices++] = (ni + 1) + nj * nx;
			indices[mNumScreenMeshIndices++] = (ni + 1) + nj * nx;
			indices[mNumScreenMeshIndices++] = ni + (nj + 1) * nx;
			indices[mNumScreenMeshIndices++] = ni + nj * nx;
		}
	}

	D3D11_BUFFER_DESC ibd;
	ibd.BindFlags = D3D11_BIND_INDEX_BUFFER;
	ibd.ByteWidth = mNumScreenMeshIndices * sizeof(UINT);
	ibd.CPUAccessFlags = 0;
	ibd.MiscFlags = 0;
	ibd.StructureByteStride = 0;
	ibd.Usage = D3D11_USAGE_IMMUTABLE;

	D3D11_SUBRESOURCE_DATA iinitData;
	iinitData.pSysMem = indices;

	mDevice->CreateBuffer(&ibd, &iinitData, &mScreenMeshIB);

	delete[] indices;
}

void WaterBruneton::getSpectrumSample(int i, int j, float lengthScale, float kMin, float * result)
{
	float dk = XM_2PI / lengthScale;
	float kx = i * dk;
	float ky = j * dk;
	if (abs(kx) < kMin && abs(ky) < kMin)
	{
		result[0] = 0.0;
		result[1] = 0.0;
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

// 1/kx and 1/ky in meters
float WaterBruneton::spectrum(float kx, float ky, bool omnispectrum)
{
	float U10 = WIND;
	float Omega = OMEGA;

	// phase speed
	float k = sqrt(kx * kx + ky * ky);
	float c = omega(k) / k;

	// spectral peak
	float kp = 9.81 * sqr(Omega / U10); // after Eq 3
	float cp = omega(kp) / kp;

	// friction velocity
	float z0 = 3.7e-5 * sqr(U10) / 9.81 * pow(U10 / cp, 0.9f); // Eq 66
	float u_star = 0.41 * U10 / log(10.0 / z0); // Eq 60

	float Lpm = exp(-5.0 / 4.0 * sqr(kp / k)); // after Eq 3
	float gamma = Omega < 1.0 ? 1.7 : 1.7 + 6.0 * log(Omega); // after Eq 3 // log10 or log??
	float sigma = 0.08 * (1.0 + 4.0 / pow(Omega, 3.0f)); // after Eq 3
	float Gamma = exp(-1.0 / (2.0 * sqr(sigma)) * sqr(sqrt(k / kp) - 1.0));
	float Jp = pow(gamma, Gamma); // Eq 3
	float Fp = Lpm * Jp * exp(-Omega / sqrt(10.0) * (sqrt(k / kp) - 1.0)); // Eq 32
	float alphap = 0.006 * sqrt(Omega); // Eq 34
	float Bl = 0.5 * alphap * cp / c * Fp; // Eq 31

	float alpham = 0.01 * (u_star < cm ? 1.0 + log(u_star / cm) : 1.0 + 3.0 * log(u_star / cm)); // Eq 44
	float Fm = exp(-0.25 * sqr(k / km - 1.0)); // Eq 41
	float Bh = 0.5 * alpham * cm / c * Fm * Lpm; // Eq 40 (fixed)

	if (omnispectrum) {
		return A * (Bl + Bh) / (k * sqr(k)); // Eq 30
	}

	float a0 = log(2.0) / 4.0; float ap = 4.0; float am = 0.13 * u_star / cm; // Eq 59
	float Delta = tanh(a0 + ap * pow(c / cp, 2.5f) + am * pow(cm / c, 2.5f)); // Eq 57

	float phi = atan2(ky, kx);

	if (kx < 0.0) {
		return 0.0;
	}
	else {
		Bl *= 2.0;
		Bh *= 2.0;
	}

	return A * (Bl + Bh) * (1.0 + Delta * cos(2.0 * phi)) / (2.0 * M_PI * sqr(sqr(k))); // Eq 67
}

float WaterBruneton::omega(float k)
{
	return sqrt(9.81 * k * (1.0 + pow(k / km, 2))); // Eq 24
}

float WaterBruneton::getSlopeVariance(float kx, float ky, float * spectrumSample)
{
	float kSquare = kx * kx + ky * ky;
	float real = spectrumSample[0];
	float img = spectrumSample[1];
	float hSquare = real * real + img * img;
	return kSquare * hSquare * 2.0f;
}
