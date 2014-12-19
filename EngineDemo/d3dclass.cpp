#include "d3dclass.h"

// ------------------------------------------------------------------------
//                           D3DClass definition
// ------------------------------------------------------------------------

D3DClass::D3DClass()
	: m4xMSAAQuality(0),

	mDevice(nullptr),
	mImmediateContext(nullptr),
	mSwapChain(nullptr),
	mDepthStencilBuffer(nullptr),
	mRenderTargetView(nullptr),
	mDepthStencilView(nullptr),

	mEnable4xMSAA(true)
{
	ZeroMemory(&mScreenViewport, sizeof(D3D11_VIEWPORT));

	// temp
	mStartIndex = 0.0f;
}

D3DClass::D3DClass(const D3DClass& other)
{

}

D3DClass::~D3DClass()
{
	Shutdown();
}

// Init device, device context, and swap chain
bool D3DClass::Init(HWND hwnd, UINT mClientWidth, UINT mClientHeight, std::shared_ptr<INIReader> &Settings)
{
	// Create device and device context.
	UINT createDeviceFlags = 0;
#if defined(DEBUG) || defined(_DEBUG)
	createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

	D3D_FEATURE_LEVEL targetFeatureLevels = D3D_FEATURE_LEVEL_11_1;
	D3D_FEATURE_LEVEL featureLevel;

	HRESULT hr = D3D11CreateDevice(0, D3D_DRIVER_TYPE_HARDWARE, 0, createDeviceFlags, &targetFeatureLevels, 1,
		D3D11_SDK_VERSION, (ID3D11Device**)&mDevice, &featureLevel, (ID3D11DeviceContext**)&mImmediateContext);
	if (FAILED(hr))
	{
		MessageBox(0, L"D3D11CreateDevice Failed.", 0, 0);
		LogError(L"D3D11CreateDevice failed. Exiting...");
		return false;
	}

	if (mDevice->GetFeatureLevel() < D3D_FEATURE_LEVEL_11_0)
	{
		MessageBox(0, L"Direct3D Feature Level 11 unsupported.", 0, 0);
		LogSuccess(L"Direct3D Feature Level 11 unsupported. Exiting...");
		return false;
	}
	else if (mDevice->GetFeatureLevel() == D3D_FEATURE_LEVEL_11_0)
	{
		MessageBox(0, L"Direct3D in feature level 11_0. Possible unexpected behavior.", 0, 0);
		LogNotice(L"Direct3D in feature level 11_0. Possible unexpected behavior.");
	}

	// Check 4xMSAA quality support for back buffer format.
	mDevice->CheckMultisampleQualityLevels(DXGI_FORMAT_R8G8B8A8_UNORM, 4, &m4xMSAAQuality);

	// Describe swap chain.
	DXGI_SWAP_CHAIN_DESC1 sd;
	sd.Width = 0;
	sd.Height = 0;
	sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	sd.Stereo = 0;

	if (mEnable4xMSAA)
	{
		sd.SampleDesc.Count = 4;
		sd.SampleDesc.Quality = m4xMSAAQuality - 1;
	}
	else
	{
		sd.SampleDesc.Count = 1;
		sd.SampleDesc.Quality = 0;
	}

	sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	sd.BufferCount = 1;
	sd.Scaling = DXGI_SCALING_STRETCH;
	sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
	sd.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
	sd.Flags = 0;

	// Get Factory used for creating device
	IDXGIDevice2 *dxgiDevice;
	mDevice->QueryInterface(__uuidof(IDXGIDevice2), (void**)&dxgiDevice);

	IDXGIAdapter *dxgiAdapter;
	dxgiDevice->GetAdapter(&dxgiAdapter);

	IDXGIFactory2 *dxgiFactory;
	dxgiAdapter->GetParent(__uuidof(IDXGIFactory2), (void**)&dxgiFactory);

	// Create swapchain
	dxgiFactory->CreateSwapChainForHwnd(mDevice, hwnd, &sd, NULL, NULL, &mSwapChain);

	ReleaseCOM(dxgiFactory);
	ReleaseCOM(dxgiAdapter);
	ReleaseCOM(dxgiDevice);

	// Rest of initialization can be done throught OnResize()
	OnResize(mClientWidth, mClientHeight);

	// temp
	return InitEVERYTHING();
	//return true;
}

// Change and recreated necessery resources after window resize
void D3DClass::OnResize(UINT mClientWidth, UINT mClientHeight)
{
	assert(mDevice);
	assert(mImmediateContext);
	assert(mSwapChain);

	// Release old views and buffers
	ReleaseCOM(mRenderTargetView);
	ReleaseCOM(mDepthStencilBuffer);
	ReleaseCOM(mDepthStencilView);

	// Resize swap chain and (re)create render target
	mSwapChain->ResizeBuffers(1, mClientWidth, mClientHeight, DXGI_FORMAT_R8G8B8A8_UNORM, 0);
	ID3D11Texture2D* backBuffer;
	mSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&backBuffer));
	mDevice->CreateRenderTargetView(backBuffer, 0, &mRenderTargetView);
	ReleaseCOM(backBuffer);

	// Create depth/stencil buffer and view
	D3D11_TEXTURE2D_DESC depthStencilDesc;

	depthStencilDesc.Width = mClientWidth;
	depthStencilDesc.Height = mClientHeight;
	depthStencilDesc.MipLevels = 1;
	depthStencilDesc.ArraySize = 1;
	depthStencilDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;

	if (mEnable4xMSAA)
	{
		depthStencilDesc.SampleDesc.Count = 4;
		depthStencilDesc.SampleDesc.Quality = m4xMSAAQuality - 1;
	}
	else
	{
		depthStencilDesc.SampleDesc.Count = 1;
		depthStencilDesc.SampleDesc.Quality = 0;
	}

	depthStencilDesc.Usage = D3D11_USAGE_DEFAULT;
	depthStencilDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
	depthStencilDesc.CPUAccessFlags = 0;
	depthStencilDesc.MiscFlags = 0;

	mDevice->CreateTexture2D(&depthStencilDesc, 0, &mDepthStencilBuffer);
	mDevice->CreateDepthStencilView(mDepthStencilBuffer, 0, &mDepthStencilView);

	// Bind to the pipeline
	mImmediateContext->OMSetRenderTargets(1, &mRenderTargetView, mDepthStencilView);

	// Set viewport
	mScreenViewport.TopLeftX = 0;
	mScreenViewport.TopLeftY = 0;
	mScreenViewport.Width = static_cast<float>(mClientWidth);
	mScreenViewport.Height = static_cast<float>(mClientHeight);
	mScreenViewport.MinDepth = 0.0f;
	mScreenViewport.MaxDepth = 1.0f;

	mImmediateContext->RSSetViewports(1, &mScreenViewport);
}

void D3DClass::Shutdown()
{
	ReleaseCOM(mRenderTargetView);
	ReleaseCOM(mDepthStencilBuffer);
	ReleaseCOM(mSwapChain);
	ReleaseCOM(mDepthStencilView);

	if (mImmediateContext) mImmediateContext->ClearState();

	ReleaseCOM(mImmediateContext);
	ReleaseCOM(mDevice);
}

// Clear back buffer and depth-stencil buffer
void D3DClass::BeginScene()
{
	mImmediateContext->ClearRenderTargetView(mRenderTargetView, reinterpret_cast<const float*>(&DirectX::Colors::Black));
	mImmediateContext->ClearDepthStencilView(mDepthStencilView, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);
}

// Present frame
void D3DClass::EndScene()
{
	mSwapChain->Present(0, 0);
}


// ------------------------------------------------------------------------
//                           D3DClass definition 
//                   temporal stuff for later refactoring
// ------------------------------------------------------------------------

bool D3DClass::Render()
{
	// view, world, projection matrixes
	XMVECTOR pos = XMVectorSet(0, 1, -1, 1.0f);
	XMVECTOR target = XMVectorSet(0, 0, 0, 1.0f);
	XMVECTOR up = XMVectorSet(0, 1, 0, 0);
	mCamera.SetPosition(mStartIndex, 0.0f, -2.0f);
	mCamera.SetLookAt(mStartIndex, 0.0f, 0.0f);

	XMMATRIX V = XMMatrixLookAtLH(pos, target, up);
	//XMStoreFloat4x4(&mView, V);

	XMMATRIX I = XMMatrixIdentity();
	XMStoreFloat4x4(&mRoadWorld, I);

	XMMATRIX P = XMMatrixPerspectiveFovLH(XM_PIDIV2, 1.6f, 1.0f, 10.0f);
	XMStoreFloat4x4(&mProj, P);

	UINT stride = sizeof(Vertex);
	UINT offset = 0;

	// set index buffer
	mImmediateContext->IASetIndexBuffer(mRoadIB, DXGI_FORMAT_R32_UINT, 0);
	// set vertex buffer
	mImmediateContext->IASetVertexBuffers(0, 1, &mRoadVB, &stride, &offset);
	// set primitive type
	mImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

	// mapping constant buffers
	D3D11_MAPPED_SUBRESOURCE mappedResources;
	MastrixBufferType *dataPtr;

	if (FAILED(mImmediateContext->Map(mMatrixBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResources))) return false;

	dataPtr = (MastrixBufferType*)mappedResources.pData;

	XMMATRIX view = mCamera.GetViewMatrix();
	
	view = XMMatrixTranspose(view);
	P = XMMatrixTranspose(P);

	dataPtr->gView = view;
	dataPtr->gProj = P;

	mImmediateContext->Unmap(mMatrixBuffer, 0);

	UINT bufferNumber = 0;

	mImmediateContext->VSSetConstantBuffers(bufferNumber, 1, &mMatrixBuffer);

	mImmediateContext->IASetInputLayout(mInputLayout);

	mImmediateContext->VSSetShader(mVertexShader, NULL, 0);
	mImmediateContext->PSSetShader(mPixelShader, NULL, 0);

	//if (mStartIndex < 0) mStartIndex = 0;
	//if (mStartIndex > 38) mStartIndex = 38;
	mImmediateContext->DrawIndexed(3, 0, 0);
	//mImmediateContext->Draw(3, 2);

	return true;
}

bool D3DClass::InitEVERYTHING()
{
	// create road vertex buffer
	mRoadVertexCount = 6;
	std::vector<Vertex> vertices(mRoadVertexCount);

	/*for (int i = 0; i < 21; ++i)
	{
		vertices[2 * i].Pos = XMFLOAT3((float)i - 10.0f, 0.0f, 5.0f);
		vertices[2*i].Color = XMFLOAT4((float)(i%4)/4.0f, (float)((i+1) % 4) / 4.0f, (float)((i+2) % 4) / 4.0f, 1.0f);
		
		vertices[2 * i + 1].Pos = XMFLOAT3((float)i - 10.0f, 0.0f, -5.0f);
		vertices[2 * i].Color = XMFLOAT4((float)((i+1) % 4) / 4.0f, (float)((i + 2) % 4) / 4.0f, (float)((i + 3) % 4) / 4.0f, 1.0f);
	}

	for (int i = 0; i < vertices.size(); ++i)
	{
		LogNotice(std::to_wstring(vertices[i].Pos.z));
	}*/

	/*for (int i = 0; i < 3; ++i)
	{
		vertices[2 * i].Pos = XMFLOAT3((float)i - 1.0f, 0.0f, 5.0f);
		vertices[2 * i].Color = XMFLOAT4((float)(i % 4) / 4.0f, (float)((i + 1) % 4) / 4.0f, (float)((i + 2) % 4) / 4.0f, 1.0f);

		vertices[2 * i + 1].Pos = XMFLOAT3((float)i - 10.0f, 0.0f, -5.0f);
		vertices[2 * i].Color = XMFLOAT4((float)((i + 1) % 4) / 4.0f, (float)((i + 2) % 4) / 4.0f, (float)((i + 3) % 4) / 4.0f, 1.0f);
	}*/
	vertices[0] = { XMFLOAT3(-1.0f, -0.5f, 2.0f), XMFLOAT4(0.0f, 1.0f, 0.0f, 1.0f) };
	vertices[1] = { XMFLOAT3(-1.0f,  0.5f, 2.0f), XMFLOAT4(0.0f, 0.0f, 1.0f, 1.0f) };
	vertices[2] = { XMFLOAT3( 1.0f, -0.5f, 2.0f), XMFLOAT4(1.0f, 0.0f, 0.0f, 1.0f) };
	vertices[3] = { XMFLOAT3( 1.0f,  0.5f, 2.0f), XMFLOAT4(1.0f, 0.0f, 0.0f, 1.0f) };
	vertices[4] = { XMFLOAT3( 1.0f, -0.5f, 7.0f), XMFLOAT4(0.0f, 1.0f, 0.0f, 1.0f) };
	vertices[5] = { XMFLOAT3( 1.0f,  0.5f, 7.0f), XMFLOAT4(0.0f, 0.0f, 1.0f, 1.0f) };


	D3D11_BUFFER_DESC vbd;
	vbd.Usage = D3D11_USAGE_IMMUTABLE;
	vbd.ByteWidth = sizeof(Vertex) * mRoadVertexCount;
	vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vbd.CPUAccessFlags = 0;
	vbd.MiscFlags = 0;
	vbd.StructureByteStride = 0;
	D3D11_SUBRESOURCE_DATA vinitData;
	vinitData.pSysMem = &vertices[0];
	mDevice->CreateBuffer(&vbd, &vinitData, &mRoadVB);

	// create road index buffer
	std::vector<UINT> indeces(42);
	for (int i = 0; i < 42; ++i) indeces[i] = i;
	mRoadIndexCount = indeces.size();

	D3D11_BUFFER_DESC ibd;
	ibd.Usage = D3D11_USAGE_IMMUTABLE;
	ibd.ByteWidth = sizeof(UINT) * indeces.size();
	ibd.BindFlags = D3D11_BIND_INDEX_BUFFER;
	ibd.CPUAccessFlags = 0;
	ibd.MiscFlags = 0;
	ibd.StructureByteStride = 0;
	D3D11_SUBRESOURCE_DATA iinitData;
	iinitData.pSysMem = &indeces[0];
	mDevice->CreateBuffer(&ibd, &iinitData, &mRoadIB);

	//initilize shaders
	ifstream vs_stream, ps_stream;
	size_t vs_size, ps_size;
	char *vs_data, *ps_data;
	int numElements;

	D3D11_BUFFER_DESC matrixBufferDesc;

	vs_stream.open("..\\Debug\\BasicVS.cso", ifstream::in | ifstream::binary);
	if (vs_stream.good())
	{
		vs_stream.seekg(0, ios::end);
		vs_size = size_t(vs_stream.tellg());
		vs_data = new char[vs_size];
		vs_stream.seekg(0, ios::beg);
		vs_stream.read(&vs_data[0], vs_size);
		vs_stream.close();

		if (FAILED(mDevice->CreateVertexShader(vs_data, vs_size, 0, &mVertexShader)))
		{
			LogError(L"Failed to create Vertex Shader");
			return false;
		}
	}
	else
	{
		LogError(L"Failed to open BasicVS.cso file");
		return false;
	}

	LogSuccess(L"Vertex Shader created.");

	ps_stream.open("..\\Debug\\BasicPS.cso", ifstream::in | ifstream::binary);
	if (ps_stream.good())
	{
		ps_stream.seekg(0, ios::end);
		ps_size = size_t(ps_stream.tellg());
		ps_data = new char[ps_size];
		ps_stream.seekg(0, ios::beg);
		ps_stream.read(&ps_data[0], ps_size);
		ps_stream.close();

		if (FAILED(mDevice->CreatePixelShader(ps_data, ps_size, 0, &mPixelShader))) return false;
	}
	else return false;

	LogSuccess(L"Pixel Shader created");

	D3D11_INPUT_ELEMENT_DESC vertexDesc[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 }
	};

	numElements = sizeof(vertexDesc) / sizeof(vertexDesc[0]);

	if (FAILED(mDevice->CreateInputLayout(vertexDesc, numElements, vs_data, vs_size, &mInputLayout))) return false;

	LogSuccess(L"Input Layout created");

	delete[] vs_data;
	delete[] ps_data;

	matrixBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
	matrixBufferDesc.ByteWidth = sizeof(MastrixBufferType);
	matrixBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	matrixBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	matrixBufferDesc.MiscFlags = 0;
	matrixBufferDesc.StructureByteStride = 0;

	if (FAILED(mDevice->CreateBuffer(&matrixBufferDesc, NULL, &mMatrixBuffer))) return false;

	LogSuccess(L"Constant Buffer created");

	return true;
}