#include "Water.h"

#include "Utilities\CreateShader.h"

WaterClass::WaterClass() :
	mQuadPatchVB(nullptr),
	mQuadPatchIB(nullptr),
	mInputLayout(nullptr),
	mVertexShader(nullptr),
	mPixelShader(nullptr),
	g(9.80665f)
{
	XMStoreFloat4x4(&mWorld, XMMatrixIdentity());
}

WaterClass::~WaterClass()
{
	ReleaseCOM(mQuadPatchIB);
	ReleaseCOM(mQuadPatchVB);

	ReleaseCOM(cbPerFrameDS);

	ReleaseCOM(mInputLayout);
	ReleaseCOM(mVertexShader);
	ReleaseCOM(mPixelShader);

	for (int i = 0; i < mComputeShader.size(); ++i) ReleaseCOM(mComputeShader[i]);

	ReleaseCOM(mRastStateFrame);
}

bool WaterClass::Init(ID3D11Device1 * device, ID3D11DeviceContext1 * dc)
{
	if (!CreateInputLayoutAndShaders(device)) return false;

	mNumPatchVertRows = 32 + 1;
	mNumPatchVertCols = 32 + 1;

	mNumPatchVertices = mNumPatchVertRows*mNumPatchVertCols;
	mNumPatchQuadFaces = (mNumPatchVertRows - 1)*(mNumPatchVertCols - 1);

	// Matrix buffers
	D3D11_BUFFER_DESC constantBufferDesc;
	constantBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
	constantBufferDesc.ByteWidth = sizeof(cbPerFrameDSType);
	constantBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	constantBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	constantBufferDesc.MiscFlags = 0;
	constantBufferDesc.StructureByteStride = 0;

	if (FAILED(device->CreateBuffer(&constantBufferDesc, NULL, &cbPerFrameDS))) return false;

	constantBufferDesc.ByteWidth = sizeof(cbPerFrameVSType);

	if (FAILED(device->CreateBuffer(&constantBufferDesc, NULL, &cbPerFrameVS))) return false;

	constantBufferDesc.ByteWidth = sizeof(TimeBufferCS);

	if (FAILED(device->CreateBuffer(&constantBufferDesc, NULL, &FFTPrepBuffer))) return false;

	constantBufferDesc.ByteWidth = sizeof(FFTParameters);

	if (FAILED(device->CreateBuffer(&constantBufferDesc, NULL, &FFTBuffer))) return false;

	constantBufferDesc.ByteWidth = sizeof(cbPerFrameHSType);

	if (FAILED(device->CreateBuffer(&constantBufferDesc, NULL, &cbPerFrameHS))) return false;

	constantBufferDesc.ByteWidth = sizeof(cbPerFramePSType);

	if (FAILED(device->CreateBuffer(&constantBufferDesc, NULL, &cbPerFramePS))) return false;
	
	N = 255;
	Nplus1 = N + 1;
	A = 0.000002f;
	w = XMFLOAT2(5.0f, 5.0f);
	length = 256.0f;
	time = 0.0f;

	vertices.resize(Nplus1*Nplus1);
	indices.reserve(Nplus1*Nplus1 * 6);

	distribution = normal_distribution<float>(0.0f, 1.0f);

	int index;

	BuildQuadPatchIB(device);
	BuildQuadPatchVB(device);

	CreateInitialDataResource(device);

	D3D11_RASTERIZER_DESC rastDesc;
	ZeroMemory(&rastDesc, sizeof(D3D11_RASTERIZER_DESC));
	rastDesc.FillMode = D3D11_FILL_WIREFRAME;
	rastDesc.CullMode = D3D11_CULL_NONE;
	rastDesc.FrontCounterClockwise = false;
	rastDesc.DepthClipEnable = false;

	if (FAILED(device->CreateRasterizerState(&rastDesc, &mRastStateFrame))) return false;

	mSamplerStates = new ID3D11SamplerState*[3];

	D3D11_SAMPLER_DESC samplerDesc;
	samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.BorderColor[0] = 0.0f;
	samplerDesc.BorderColor[1] = 0.0f;
	samplerDesc.BorderColor[2] = 0.0f;
	samplerDesc.BorderColor[3] = 0.0f;
	samplerDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
	samplerDesc.MaxAnisotropy = 1;
	samplerDesc.MaxLOD = FLT_MAX;
	samplerDesc.MinLOD = -FLT_MAX;
	samplerDesc.MipLODBias = 0;

	samplerDesc.Filter = D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT;
	samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;

	if (FAILED(device->CreateSamplerState(&samplerDesc, &mSamplerStates[0]))) return false;

	samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;

	if (FAILED(device->CreateSamplerState(&samplerDesc, &mSamplerStates[1]))) return false;

	samplerDesc.Filter = D3D11_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
	samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_BORDER;
	samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_BORDER;
	samplerDesc.ComparisonFunc = D3D11_COMPARISON_LESS;

	if (FAILED(device->CreateSamplerState(&samplerDesc, &mSamplerStates[2]))) return false;

	return true;
}

void WaterClass::Draw(ID3D11DeviceContext1 * mImmediateContext, std::shared_ptr<CameraClass> Camera, DirectionalLight& light, ID3D11ShaderResourceView * ShadowMap)
{
	XMMATRIX ShadowViewProjTrans = light.GetViewProjTrans();
	XMMATRIX ShadowMapProjTrans = light.GetMapProjTrans();

	XMMATRIX ViewProjTrans = Camera->GetViewProjTransMatrix();

	UINT stride = sizeof(Vertex);
	UINT offset = 0;

	XMFLOAT4 worldPlanes[6];
	ExtractFrustrumPlanes(worldPlanes, Camera->GetViewProjMatrix());

	// PS
	mImmediateContext->PSSetShader(mPixelShader, NULL, 0);

	// DS
	D3D11_MAPPED_SUBRESOURCE mappedResources;
	cbPerFrameDSType* dataPtrDS;

	mImmediateContext->Map(cbPerFrameDS, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResources);
	dataPtrDS = (cbPerFrameDSType*)mappedResources.pData;

	dataPtrDS->gViewProj = Camera->GetViewProjTransMatrix();
	dataPtrDS->gShadowTrans = Camera->GetViewProjTransMatrix();

	mImmediateContext->Unmap(cbPerFrameDS, 0);

	mImmediateContext->DSSetConstantBuffers(0, 1, &cbPerFrameDS);
	mImmediateContext->DSSetShaderResources(0, 1, &mFFTSRV[1][0]);

	mImmediateContext->DSSetSamplers(0, 1, &mSamplerStates[1]);

	mImmediateContext->DSSetShader(mDomainShader, NULL, 0);

	// HS
	cbPerFrameHSType* dataPtrHS;
	mImmediateContext->Map(cbPerFrameHS, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResources);
	dataPtrHS = (cbPerFrameHSType*)mappedResources.pData;

	XMStoreFloat3(&(dataPtrHS->gEyePosW), Camera->GetPosition());
	dataPtrHS->gMaxDist = 500.0f;
	dataPtrHS->gMinDist = 20.0f;
	dataPtrHS->gMaxTess = 6.0f;
	dataPtrHS->gMinTess = 0.0f;
	for (int i = 0; i < 6; ++i)
		dataPtrHS->gWorldFrustumPlanes[i] = worldPlanes[i];
	dataPtrHS->gFrustumCull = 1;

	mImmediateContext->Unmap(cbPerFrameHS, 0);

	mImmediateContext->HSSetConstantBuffers(0, 1, &cbPerFrameHS);

	mImmediateContext->HSSetShader(mHullShader, NULL, 0);

	// VS
	cbPerFrameVSType *dataPtr;
	mImmediateContext->Map(cbPerFrameVS, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResources);

	dataPtr = (cbPerFrameVSType*)mappedResources.pData;

	dataPtr->gWorld = XMMatrixTranspose(XMMatrixTranslation(0.0f, 10.0f, 0.0f));

	mImmediateContext->Unmap(cbPerFrameVS, 0);

	mImmediateContext->VSSetConstantBuffers(0, 1, &cbPerFrameVS);

	mImmediateContext->VSSetShaderResources(0, 1, &mFFTSRV[1][0]);

	mImmediateContext->VSSetSamplers(0, 1, &mSamplerStates[0]);

	mImmediateContext->VSSetShader(mVertexShader, NULL, 0);

	// IA
	mImmediateContext->IASetInputLayout(mInputLayout);

	mImmediateContext->IASetIndexBuffer(mQuadPatchIB, DXGI_FORMAT_R32_UINT, 0);
	mImmediateContext->IASetVertexBuffers(0, 1, &mQuadPatchVB, &stride, &offset);
	mImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_4_CONTROL_POINT_PATCHLIST);

	mImmediateContext->RSSetState(mRastStateFrame);

	mImmediateContext->DrawIndexed(indices_count, 0, 0);

	mImmediateContext->HSSetShader(0, 0, 0);
	mImmediateContext->DSSetShader(0, 0, 0);

	ID3D11ShaderResourceView* ppSRVNULL = NULL;
	mImmediateContext->VSSetShaderResources(0, 1, &ppSRVNULL);
	mImmediateContext->DSSetShaderResources(0, 1, &ppSRVNULL);
}

void WaterClass::evaluateWavesGPU(float t, ID3D11DeviceContext1 * mImmediateContext)
{
	time += t;

	ID3D11UnorderedAccessView* ppUAViewNULL[2] = { NULL, NULL };
	ID3D11ShaderResourceView* ppSRVNULL[2] = { NULL, NULL };

	mImmediateContext->CSSetShader(mComputeShader[0], NULL, 0);
	mImmediateContext->CSSetShaderResources(0, 1, &mFFTInitialSRV);

	mImmediateContext->CSSetUnorderedAccessViews(0, 2, mFFTUAV[0], NULL);

	D3D11_MAPPED_SUBRESOURCE mappedResource;
	TimeBufferCS* timePtr;

	mImmediateContext->Map(FFTPrepBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	timePtr = (TimeBufferCS*)mappedResource.pData;

	timePtr->time = XMFLOAT4(time, time, time, time);

	mImmediateContext->Unmap(FFTPrepBuffer, 0);

	mImmediateContext->CSSetConstantBuffers(0, 1, &FFTPrepBuffer);

	mImmediateContext->Dispatch(1, 256, 1);


	mImmediateContext->CSSetUnorderedAccessViews(0, 2, ppUAViewNULL, NULL);
	mImmediateContext->CSSetShaderResources(0, 2, ppSRVNULL);
	
	mImmediateContext->CSSetShader(mComputeShader[1], NULL, 0);
	mImmediateContext->CSSetShaderResources(0, 2, mFFTSRV[0]);
	mImmediateContext->CSSetUnorderedAccessViews(0, 2, mFFTUAV[1], NULL);

	FFTParameters* fftParams;
	mImmediateContext->Map(FFTBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	fftParams = (FFTParameters*)mappedResource.pData;

	fftParams->col_row_pass[0] = 0;

	mImmediateContext->Unmap(FFTBuffer, 0);

	mImmediateContext->CSSetConstantBuffers(0, 1, &FFTBuffer);

	mImmediateContext->Dispatch(1, 256, 1);


	mImmediateContext->CSSetUnorderedAccessViews(0, 2, ppUAViewNULL, NULL);
	mImmediateContext->CSSetShaderResources(0, 2, ppSRVNULL);

	mImmediateContext->CSSetShaderResources(0, 2, mFFTSRV[1]);
	mImmediateContext->CSSetUnorderedAccessViews(0, 2, mFFTUAV[0], NULL);

	mImmediateContext->Map(FFTBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	fftParams = (FFTParameters*)mappedResource.pData;

	fftParams->col_row_pass[0] = 1;

	mImmediateContext->Unmap(FFTBuffer, 0);

	mImmediateContext->CSSetConstantBuffers(0, 1, &FFTBuffer);

	mImmediateContext->Dispatch(1, 256, 1);


	mImmediateContext->CSSetUnorderedAccessViews(0, 2, ppUAViewNULL, NULL);
	mImmediateContext->CSSetShaderResources(0, 2, ppSRVNULL);

	mImmediateContext->CSSetShader(mComputeShader[2], NULL, 0);
	mImmediateContext->CSSetShaderResources(0, 1, &mFFTSRV[0][0]);
	mImmediateContext->CSSetUnorderedAccessViews(0, 1, &mFFTUAV[1][0], NULL);

	mImmediateContext->Dispatch(1, 256, 1);
	
	mImmediateContext->CSSetShader(NULL, NULL, 0);
	mImmediateContext->CSSetUnorderedAccessViews(0, 2, ppUAViewNULL, NULL);
	mImmediateContext->CSSetShaderResources(0, 2, ppSRVNULL);
}

void WaterClass::BuildQuadPatchVB(ID3D11Device1 * device)
{
	vector<Vertex> patchVertices(mNumPatchVertCols*mNumPatchVertRows);

	float halfWidth = 512.0f;
	float halfDepth = 512.0f;

	float patchWidth = 1024.0f / (mNumPatchVertCols - 1);
	float patchDepth = 1024.0f / (mNumPatchVertRows - 1);

	for (int i = 0; i < mNumPatchVertRows; ++i)
	{
		float z = halfDepth - i*patchDepth;
		for (int j = 0; j < mNumPatchVertCols; ++j)
		{
			float x = -halfWidth + j*patchWidth;

			patchVertices[i*mNumPatchVertCols + j] = Vertex{ XMFLOAT3(x, 0.0f, z), XMFLOAT2(x / 256.0f, z / 256.0f) };
		}
	}

	D3D11_BUFFER_DESC vbd;
	vbd.Usage = D3D11_USAGE_IMMUTABLE;
	vbd.ByteWidth = sizeof(Vertex)*patchVertices.size();
	vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vbd.CPUAccessFlags = 0;
	vbd.MiscFlags = 0;
	vbd.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA vinitData;
	vinitData.pSysMem = &patchVertices[0];
	device->CreateBuffer(&vbd, &vinitData, &mQuadPatchVB);
}

bool WaterClass::BuildQuadPatchIB(ID3D11Device1 * device)
{
	vector<UINT> indices(mNumPatchQuadFaces * 4);

	int k = 0;
	for (int i = 0; i < mNumPatchVertRows - 1; ++i)
	{
		for (int j = 0; j < mNumPatchVertCols - 1; ++j)
		{
			indices[k++] = i*mNumPatchVertCols + j;
			indices[k++] = i*mNumPatchVertCols + j + 1;
			indices[k++] = (i + 1)*mNumPatchVertCols + j;
			indices[k++] = (i + 1)*mNumPatchVertCols + j + 1;
		}
	}

	indices_count = indices.size();

	D3D11_BUFFER_DESC ibd;
	ibd.Usage = D3D11_USAGE_IMMUTABLE;
	ibd.ByteWidth = sizeof(UINT)*indices.size();
	ibd.BindFlags = D3D11_BIND_INDEX_BUFFER;
	ibd.CPUAccessFlags = 0;
	ibd.MiscFlags = 0;
	ibd.StructureByteStride = 0;

	D3D11_SUBRESOURCE_DATA iinitData;
	iinitData.pSysMem = &indices[0];
	if (FAILED(device->CreateBuffer(&ibd, &iinitData, &mQuadPatchIB))) return false;

	return true;
}

bool WaterClass::CreateInputLayoutAndShaders(ID3D11Device1 * device)
{
	// compute
	ID3D11ComputeShader* tempCShader;
	CreateCSFromFile(L"..\\Debug\\WaterCS_FFTPrep.cso", device, tempCShader);
	mComputeShader.push_back(tempCShader);

	CreateCSFromFile(L"..\\Debug\\WaterCS_FFT.cso", device, tempCShader);
	mComputeShader.push_back(tempCShader);

	CreateCSFromFile(L"..\\Debug\\WaterCS_FFTPost.cso", device, tempCShader);
	mComputeShader.push_back(tempCShader);
	
	// pixel
	CreatePSFromFile(L"..\\Debug\\WaterPS.cso", device, mPixelShader);

	// domain
	CreateDSFromFile(L"..\\Debug\\WaterDS.cso", device, mDomainShader);
	
	// hull
	CreateHSFromFile(L"..\\Debug\\WaterHS.cso", device, mHullShader);
	
	// vertex and input layout
	D3D11_INPUT_ELEMENT_DESC vertexDesc[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 }
	};

	int numElements = sizeof(vertexDesc) / sizeof(vertexDesc[0]);

	CreateVSAndInputLayout(L"..\\Debug\\WaterVS.cso", device, mVertexShader, vertexDesc, numElements, mInputLayout);

	return true;
}

bool WaterClass::CreateInitialDataResource(ID3D11Device1 * device)
{
	vector<FFTInitialType> Constants;
	complex<float> h_0, h_0conj;
	float w;
	for (int i = 0; i < Nplus1; ++i)
	{
		for (int j = 0; j < Nplus1; ++j)
		{
			h_0 = hTilde_0(i, j);
			h_0conj = conj(hTilde_0(-i, -j));
			w = Dispersion(i, j);

			Constants.push_back(FFTInitialType{ XMFLOAT2(real(h_0), imag(h_0)), XMFLOAT2(real(h_0conj), imag(h_0conj)), w });
		}
	}

	D3D11_BUFFER_DESC fftDesc = {};
	fftDesc.Usage = D3D11_USAGE_IMMUTABLE;
	fftDesc.ByteWidth = sizeof(FFTInitialType)*Constants.size();
	fftDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	fftDesc.StructureByteStride = sizeof(FFTInitialType);
	fftDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;

	D3D11_SUBRESOURCE_DATA fftData = {};
	fftData.pSysMem = &Constants[0];
	HRESULT hr = device->CreateBuffer(&fftDesc, &fftData, &mFFTInitial);

	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Format = DXGI_FORMAT_UNKNOWN;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
	srvDesc.BufferEx.FirstElement = 0;
	srvDesc.BufferEx.Flags = 0;
	srvDesc.BufferEx.NumElements = Constants.size();

	hr = device->CreateShaderResourceView(mFFTInitial, &srvDesc, &mFFTInitialSRV);

	D3D11_TEXTURE2D_DESC FFTRIDesc;
	FFTRIDesc.Width = Nplus1;
	FFTRIDesc.Height = Nplus1;
	FFTRIDesc.MipLevels = 1;
	FFTRIDesc.ArraySize = 1;
	FFTRIDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	FFTRIDesc.SampleDesc = { 1, 0 };
	FFTRIDesc.Usage = D3D11_USAGE_DEFAULT;
	FFTRIDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	FFTRIDesc.CPUAccessFlags = 0;
	FFTRIDesc.MiscFlags = 0;

	ID3D11Texture2D *FFTReal0 = 0, *FFTImag0 = 0, *FFTReal1 = 0, *FFTImag1 = 0;
	hr = device->CreateTexture2D(&FFTRIDesc, 0, &FFTReal0);
	hr = device->CreateTexture2D(&FFTRIDesc, 0, &FFTImag0);
	hr = device->CreateTexture2D(&FFTRIDesc, 0, &FFTReal1);
	hr = device->CreateTexture2D(&FFTRIDesc, 0, &FFTImag1);

	D3D11_SHADER_RESOURCE_VIEW_DESC FFTRISRVDesc;
	FFTRISRVDesc.Format = FFTRIDesc.Format;
	FFTRISRVDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	FFTRISRVDesc.Texture2D.MostDetailedMip = 0;
	FFTRISRVDesc.Texture2D.MipLevels = FFTRIDesc.MipLevels;

	hr = device->CreateShaderResourceView(FFTReal0, &FFTRISRVDesc, &mFFTSRV[0][0]);
	hr = device->CreateShaderResourceView(FFTImag0, &FFTRISRVDesc, &mFFTSRV[0][1]);
	hr = device->CreateShaderResourceView(FFTReal1, &FFTRISRVDesc, &mFFTSRV[1][0]);
	hr = device->CreateShaderResourceView(FFTImag1, &FFTRISRVDesc, &mFFTSRV[1][1]);
	
	D3D11_UNORDERED_ACCESS_VIEW_DESC FFTRIUAVDesc = {};
	FFTRIUAVDesc.Format = FFTRIDesc.Format;
	FFTRIUAVDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
	FFTRIUAVDesc.Texture2DArray.MipSlice = 0;

	hr = device->CreateUnorderedAccessView(FFTReal0, &FFTRIUAVDesc, &mFFTUAV[0][0]);
	hr = device->CreateUnorderedAccessView(FFTImag0, &FFTRIUAVDesc, &mFFTUAV[0][1]);
	hr = device->CreateUnorderedAccessView(FFTReal1, &FFTRIUAVDesc, &mFFTUAV[1][0]);
	hr = device->CreateUnorderedAccessView(FFTImag1, &FFTRIUAVDesc, &mFFTUAV[1][1]);

	return false;
}

float WaterClass::PhillipsSpectrum(int n, int m)
{
	XMFLOAT2 k(XM_PI*(2 * n - N) / length, XM_PI*(2 * m - N) / length);
	float k_length = sqrt(k.x*k.x + k.y*k.y);
	if (k_length < 0.000001) return 0.0f;

	float w_length = sqrt(w.x*w.x + w.y*w.y);

	float k_dot_w = (k.x / k_length)*(w.x / w_length) + (k.y / k_length)*(w.y / w_length);

	float L = w_length*w_length / g;

	float damping = 0.001f;
	float l = L*damping;

	return A*exp(-1.0f / (k_length*k_length*L*L)) / pow(k_length, 4)*pow(k_dot_w, 2)*exp(-k_length*k_length*l*l);
}

complex<float> WaterClass::hTilde_0(int n, int m)
{
	float x1, x2, w;
	do {
		x1 = 2.0f*distribution(generator) - 1.0f;
		x2 = 2.0f*distribution(generator) - 1.0f;
		w = x1*x1 + x2*x2;
	} while (w >= 1.0f);
	w = sqrt((-1.0f / log(w)) / w);
	complex<float> r = { x1*w, x2*w };
	return r * sqrt(PhillipsSpectrum(n, m) / 2.0f);
}

float WaterClass::Dispersion(int n, int m)
{
	float w_0 = XM_2PI / 200.0f;
	float kx = XM_PI*(2.0f*n - N) / length;
	float kz = XM_PI*(2.0f*m - N) / length;
	return floor(sqrt(g*sqrt(kx*kx + kz*kz)) / w_0)*w_0;
}