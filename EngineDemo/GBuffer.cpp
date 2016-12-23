#include "GBuffer.h"

#include "Utilities\RenderViewTargetStack.h"

using namespace std;
using namespace DirectX;

GBufferClass::GBufferClass()
{
}

int GBufferClass::Init(ID3D11Device1 * device, int width, int height)
{

	EXIT_ON_NULL(mGBuffer[0] =
				 TextureFactory::CreateTexture(D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, 
											   DXGI_FORMAT_R8G8B8A8_UNORM, width, height));

	EXIT_ON_NULL(mGBuffer[1] =
				 TextureFactory::CreateTexture(D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
											   DXGI_FORMAT_R16G16_FLOAT, width, height));

	mGBufferRTV[0] = mGBuffer[0]->GetRTV();
	mGBufferRTV[1] = mGBuffer[1]->GetRTV();

	// depth stencil
	D3D11_TEXTURE2D_DESC textDesc;

	ZeroMemory(&textDesc, sizeof(textDesc));

	textDesc.Width = width;
	textDesc.Height = height;
	textDesc.MipLevels = 1;
	textDesc.ArraySize = 1;
	textDesc.Format = DXGI_FORMAT_R24G8_TYPELESS;
	textDesc.SampleDesc = { 1, 0 };
	textDesc.Usage = D3D11_USAGE_DEFAULT;
	textDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
	textDesc.CPUAccessFlags = 0;
	textDesc.MiscFlags = 0;

	EXIT_ON_NULL(mDepthStencil =
		TextureFactory::CreateTexture(textDesc, { DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_R24_UNORM_X8_TYPELESS, DXGI_FORMAT_UNKNOWN }));
	
	// view port
	mViewPort = {};
	mViewPort.Width = static_cast<float>(width);
	mViewPort.Height = static_cast<float>(height);
	mViewPort.MaxDepth = 1.0;

	return S_OK;
}

void GBufferClass::Shutdown()
{
}

void GBufferClass::SetBufferRTV(ID3D11DeviceContext1 * mImmediateContext) const
{
	RenderTargetStack::Push(mImmediateContext, mGBufferRTV, mDepthStencil->GetDSV(), buffer_count);

	float colour[4] = {};
	mImmediateContext->ClearRenderTargetView(mGBuffer[0]->GetRTV(), colour);
	mImmediateContext->ClearRenderTargetView(mGBuffer[1]->GetRTV(), colour);
	mImmediateContext->ClearDepthStencilView(mDepthStencil->GetDSV(), D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 0.0f, 0);
}

void GBufferClass::UnsetBufferRTV(ID3D11DeviceContext1 * mImmediateContext) const
{
	RenderTargetStack::Pop(mImmediateContext);
}

void GBufferClass::SetBufferSRV(ID3D11DeviceContext1 * mImmediateContext, int first_slot)
{
	mImmediateContext->PSSetShaderResources(first_slot, buffer_count, mGBufferSRV);
	srv_slot = first_slot;
}

void GBufferClass::UnsetBufferSRV(ID3D11DeviceContext1 * mImmediateContext) const
{
	if (srv_slot >= 0)
	{
		ID3D11ShaderResourceView* ppSRVNULL[] = { nullptr, nullptr };
		mImmediateContext->PSSetShaderResources(srv_slot, buffer_count, ppSRVNULL);
	}
}
