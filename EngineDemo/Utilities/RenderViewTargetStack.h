#pragma once

#include <stack>
#include <utility>

#pragma comment(lib,  "d3d11.lib")

#include <d3d11_1.h>

class RenderTargetStack
{
public:
	static bool Empty()
	{
		return !Targetviews.size();
	}
	
	static bool Push(_In_ ID3D11DeviceContext1* mImmediateContext, _In_reads_(number) ID3D11RenderTargetView* const* targetviews, _In_ ID3D11DepthStencilView* depthstencil, _In_opt_ UINT number = 1)
	{
		Targetviews.push({ number, targetviews });
		DepthStencilDSV.push(depthstencil);

		mImmediateContext->OMSetRenderTargets(number, targetviews, depthstencil);

		return true;
	}

	static bool Swap(_In_ ID3D11DeviceContext1* mImmediateContext, _In_reads_(number) ID3D11RenderTargetView** targetviews, _In_ ID3D11DepthStencilView* depthstencil, _In_opt_ UINT number = 1)
	{
		while (Targetviews.size())
		{
			Targetviews.pop();
			DepthStencilDSV.pop();
		}

		Push(mImmediateContext, targetviews, depthstencil, number);

		return true;
	}

	static bool Pop(_In_ ID3D11DeviceContext1* mImmediateContext)
	{
		switch (Targetviews.size())
		{
			case 0: return true;
			case 1: return false;
			default:
				Targetviews.pop();
				DepthStencilDSV.pop();

				mImmediateContext->OMSetRenderTargets(std::get<UINT>(Targetviews.top()), std::get<ID3D11RenderTargetView* const*>(Targetviews.top()), DepthStencilDSV.top());

				return true;
		}		
	}

	static bool Update(_In_ ID3D11DeviceContext1* mImmediateContext, _In_reads_(number) ID3D11RenderTargetView** targetviews, _In_ ID3D11DepthStencilView* depthstencil, _In_opt_ UINT number = 1)
	{
		if (Targetviews.size() && std::get<ID3D11RenderTargetView* const*>(Targetviews.top()) == targetviews && DepthStencilDSV.top() == depthstencil)
		{
			mImmediateContext->OMSetRenderTargets(number, targetviews, depthstencil);
			return true;
		}
		return false;
	}
private:
	static std::stack< std::pair<UINT, ID3D11RenderTargetView* const*> > Targetviews;
	static std::stack< ID3D11DepthStencilView* > DepthStencilDSV;
};

class ViewportStack
{
public:
	static bool Empty()
	{
		return !Viewports.size();
	}

	static bool Push(_In_ ID3D11DeviceContext1* mImmediateContext, _In_reads_(number) D3D11_VIEWPORT* viewports, _In_opt_ UINT number = 1)
	{
		Viewports.push({ number, viewports });

		mImmediateContext->RSSetViewports(number, viewports);

		return true;
	}

	static bool Swap(_In_ ID3D11DeviceContext1* mImmediateContext, _In_reads_(number) D3D11_VIEWPORT* viewports, _In_opt_ UINT number = 1)
	{
		while (Viewports.size())
		{
			Viewports.pop();
		}

		Push(mImmediateContext, viewports, number);

		return true;
	}

	static bool Pop(_In_ ID3D11DeviceContext1* mImmediateContext)
	{
		switch (Viewports.size())
		{
			case 0: return true;
			case 1: return false;
			default:
				Viewports.pop();

				mImmediateContext->RSSetViewports(std::get<UINT>(Viewports.top()), std::get<D3D11_VIEWPORT*>(Viewports.top()));

				return true;
		}		
	}

	static bool Update(_In_ ID3D11DeviceContext1* mImmediateContext, _In_reads_(number) D3D11_VIEWPORT* viewports, _In_opt_ UINT number = 1)
	{
		if (Empty() && std::get<D3D11_VIEWPORT*>(Viewports.top()) == viewports)
		{
			mImmediateContext->RSSetViewports(std::get<UINT>(Viewports.top()), std::get<D3D11_VIEWPORT*>(Viewports.top()));
			return true;
		}

		return false;
	}
private:
	static std::stack< std::pair<UINT, D3D11_VIEWPORT*> > Viewports;
};