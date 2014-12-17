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

#include <d3d11_1.h>

#pragma warning(disable:4838)
#include <DirectXMath.h>
#pragma warning(default:4838)

#include <DirectXColors.h>

#include "inih\cpp\INIReader.h"
#include "loggerclass.h"

/*
	Main DirectX 3D class:
		- Create device, context device, and other render specyfic subsystems
		- Manage rendering process
*/
class D3DClass : public HasLogger
{
	public:
		D3DClass();
		D3DClass(const D3DClass &other);
		~D3DClass();

		// Init devices and subsystems
		bool Init(HWND hwnd, UINT mClientWidth, UINT mClientHeight, std::shared_ptr<INIReader> &Settings);
		
		// Change and recreate neccessery resources after window's resize
		void OnResize(UINT mClientWidht, UINT mClientHeight);

		// Close everything
		void Shutdown();

		// Render stages
		void BeginScene();
		void EndScene();

	private:
		// Setting
		bool mEnable4xMSAA;
		UINT m4xMSAAQuality; // Maximum available 4xMSAA quality

		// Subsystems and resources
		ID3D11Device1* mDevice;
		ID3D11DeviceContext1* mImmediateContext;
		IDXGISwapChain1* mSwapChain;
		ID3D11Texture2D* mDepthStencilBuffer;
		ID3D11RenderTargetView* mRenderTargetView;
		ID3D11DepthStencilView* mDepthStencilView;
		D3D11_VIEWPORT mScreenViewport;
};