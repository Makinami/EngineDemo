#pragma once

/* temp */
//---------------------------------------------------------------------------------------
// Convenience macro for releasing COM objects.
//---------------------------------------------------------------------------------------

#define ReleaseCOM(x) { if(x){ x->Release(); x = 0; } }

//---------------------------------------------------------------------------------------
// Convenience macro for deleting objects.
//---------------------------------------------------------------------------------------

#define SafeDelete(x) { delete x; x = 0; }

#pragma comment(lib,  "d3d11.lib")

#include <memory>

#include <d3d11.h>
#include <DirectXMath.h>
#include <DirectXColors.h>

#include "inih\cpp\INIReader.h"

class D3DClass
{
	public:
		D3DClass();
		D3DClass(const D3DClass &other);
		~D3DClass();

		bool Init(HWND hwnd, UINT mClientWidth, UINT mClientHeight, std::shared_ptr<INIReader> &Settings);
		void OnResize(UINT mClientWidht, UINT mClientHeight);
		void Shutdown();

		void BeginScene();
		void EndScene();

	private:
		UINT m4xMSAAQuality;

		ID3D11Device* mDevice;
		ID3D11DeviceContext* mImmediateContext;
		IDXGISwapChain* mSwapChain;
		ID3D11Texture2D* mDepthStencilBuffer;
		ID3D11RenderTargetView* mRenderTargetView;
		ID3D11DepthStencilView* mDepthStencilView;
		D3D11_VIEWPORT mScreenViewport;
		
		bool mEnable4xMSAA;
};