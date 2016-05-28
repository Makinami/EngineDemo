#include "d3dclass.h"

std::stack< std::pair<UINT, ID3D11RenderTargetView**> > RenderTargetStack::Targetviews;
std::stack< ID3D11DepthStencilView** > RenderTargetStack::DepthStencilDSV;

std::stack< std::pair<UINT, D3D11_VIEWPORT*> > ViewportStack::Viewports;

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

	mEnable4xMSAA(true),

	mRenderHeight(0),
	mRenderWidth(0)
{
	ZeroMemory(&mScreenViewport, sizeof(D3D11_VIEWPORT));

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
		return;
	}

#if defined(DEBUG) || defined(_DEBUG)
	mDevice->QueryInterface(IID_PPV_ARGS(&mDebug));
#endif

	if (mDevice->GetFeatureLevel() < D3D_FEATURE_LEVEL_11_0)
	{
		MessageBox(0, L"Direct3D Feature Level 11 unsupported.", 0, 0);
		LogSuccess(L"Direct3D Feature Level 11 unsupported. Exiting...");
		ReleaseCOM(mImmediateContext);
		ReleaseCOM(mDevice);
		return;
	}
	else if (mDevice->GetFeatureLevel() == D3D_FEATURE_LEVEL_11_0)
	{
		MessageBox(0, L"Direct3D in feature level 11_0. Possible unexpected behavior.", 0, 0);
		LogNotice(L"Direct3D in feature level 11_0. Possible unexpected behavior.");
	}
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
	if (!mDevice || !mImmediateContext)
	{
		MessageBox(0, L"D3D Init failed. Device is not created.", 0, 0);
		LogError(L"D3D Init failed. Device is not created.");
		return false;
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

	return true;
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

	// Set client size
	mRenderWidth = mClientWidth;
	mRenderHeight = mClientHeight;

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

	if (RenderTargetStack::Empty()) RenderTargetStack::Push(mImmediateContext, &mRenderTargetView, &mDepthStencilView);
	else RenderTargetStack::Update(mImmediateContext, &mRenderTargetView, &mDepthStencilView);

	// Set viewport
	mScreenViewport.TopLeftX = 0;
	mScreenViewport.TopLeftY = 0;
	mScreenViewport.Width = static_cast<float>(mClientWidth);
	mScreenViewport.Height = static_cast<float>(mClientHeight);
	mScreenViewport.MinDepth = 0.0f;
	mScreenViewport.MaxDepth = 1.0f;

	if (ViewportStack::Empty()) ViewportStack::Push(mImmediateContext, &mScreenViewport);
	else ViewportStack::Update(mImmediateContext, &mScreenViewport);
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
#if defined(DEBUG) || defined(_DEBUG)
	if (mDebug)
	{
		Sleep(500);
		mDebug->ReportLiveDeviceObjects(D3D11_RLDO_DETAIL);
		ReleaseCOM(mDebug);
	}
#endif
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

ID3D11Device1* D3DClass::GetDevice() const
{
	return mDevice;
}

ID3D11DeviceContext1* D3DClass::GetDeviceContext() const
{
	return mImmediateContext;
}