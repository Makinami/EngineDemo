#include "shadowmapclass.h"

#include "Utilities\RenderViewTargetStack.h"

ShadowMapClass::ShadowMapClass(ID3D11Device1 * device, UINT width, UINT height)
	: mWidth(width), mHeight(height), mDepthMapDSV(0), mDepthMapSRV(0)
{
	mViewport = {};
	mViewport.Width = static_cast<float>(width);
	mViewport.Height = static_cast<float>(height);
	mViewport.MaxDepth = 1.0f;

	D3D11_TEXTURE2D_DESC texDesc = {};
	texDesc.Width = mWidth;
	texDesc.Height = mHeight;
	texDesc.MipLevels = 1;
	texDesc.ArraySize = 1;
	texDesc.Format = DXGI_FORMAT_R24G8_TYPELESS;
	texDesc.SampleDesc = { 1, 0 };
	texDesc.Usage = D3D11_USAGE_DEFAULT;
	texDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
	texDesc.CPUAccessFlags = 0;
	texDesc.MiscFlags = 0;

	ID3D11Texture2D* depthMap = 0;
	device->CreateTexture2D(&texDesc, 0, &depthMap);

	D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc = {};
	dsvDesc.Flags = 0;
	dsvDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	dsvDesc.Texture2D.MipSlice = 0;
	device->CreateDepthStencilView(depthMap, &dsvDesc, &mDepthMapDSV);

	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MipLevels = texDesc.MipLevels;
	srvDesc.Texture2D.MostDetailedMip = 0;
	if (FAILED(device->CreateShaderResourceView(depthMap, &srvDesc, &mDepthMapSRV)))
	{
		int i = 0;
	}

	ReleaseCOM(depthMap);
}

ShadowMapClass::~ShadowMapClass()
{
	ReleaseCOM(mDepthMapDSV);
	ReleaseCOM(mDepthMapSRV);
}

ID3D11ShaderResourceView * ShadowMapClass::DepthMapSRV()
{
	return mDepthMapSRV;
}

void ShadowMapClass::BindDsvAndSetNullRenderTarget(ID3D11DeviceContext1 * dc)
{
	ViewportStack::Push(dc, &mViewport);

	ID3D11RenderTargetView* renderTargets[1] = { 0 };
	RenderTargetStack::Push(dc, renderTargets, mDepthMapDSV);

	dc->ClearDepthStencilView(mDepthMapDSV, D3D11_CLEAR_DEPTH, 0.0f, 0);
}

void ShadowMapClass::ClearDepthMap(ID3D11DeviceContext1 * dc)
{
	dc->ClearDepthStencilView(mDepthMapDSV, D3D11_CLEAR_DEPTH, 0.0f, 0);
}