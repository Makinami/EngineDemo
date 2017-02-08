#pragma once

#include <d3d11_1.h>

#define ReleaseCOM(x) { if(x){ x->Release(); x = nullptr; } }
#define SafeDelete(x) { delete x; x = nullptr; }

#pragma warning(disable:4838)
#include <DirectXMath.h>
#pragma warning(default:4838)

#include <wrl\client.h>

#define EXIT_ON_FAILURE(fnc)  \
	{ \
		HRESULT result; \
		if (FAILED(result = fnc)) { \
			return result; \
		} \
	}


#include "terrain.h"
#include "Utilities\CreateShader.h"
#include "Utilities\Texture.h"

#include "ResizeEvent.h"

namespace PostFX
{
	struct VertexType
	{
		XMFLOAT2 pos;
		XMFLOAT2 tex;
	};

	class Canvas : OnResizeListener
	{
	public:
		bool Init(ID3D11Device1* device, int width, int height);
		void Shutdown();

	public:
		void StartRegister(ID3D11DeviceContext1* mImmediateContext, bool clear = true) const;
		void StopRegister(ID3D11DeviceContext1* mImmediateContext) const;

		void Swap();

		void CopyDepth(ID3D11DeviceContext1* mImmediateContext);
		void CopyFrame(ID3D11DeviceContext1* mImmediateContext);
		ID3D11ShaderResourceView* const * GetDepthCopySRV() const;

		void Present(ID3D11DeviceContext1*& mImmediateContext);

	public:
		ID3D11ShaderResourceView*const* GetAddressOfSRV(bool secondary = false) const;
		ID3D11UnorderedAccessView*const* GetAddressOfUAV(bool secondary = false) const;
		ID3D11ShaderResourceView*const* GetDepthStencilSRV() const;

	private:
		HRESULT OnResize(ID3D11Device1* device, int renderWidth, int renderHeight);

		std::unique_ptr<Texture> mMain;
		std::unique_ptr<Texture> mSecondary;

		std::unique_ptr<Texture> mDepthStencilMain;
		std::unique_ptr<Texture> mDepthStencilSecondary;
		D3D11_VIEWPORT mViewPort;

		// TEMP
		ID3D11Buffer* mScreenQuadVB;
		ID3D11Buffer* mScreenQuadIB;

		ID3D11InputLayout* mDebugIL;
		ID3D11VertexShader* mDebugVS;
		ID3D11PixelShader* mDebugPS;

		struct MatrixBufferType
		{
			XMMATRIX gWorldProj;
		};

		ID3D11Buffer* MatrixBuffer;
	};

	class HDR : OnResizeListener
	{
	public:
		HDR();
		HDR(const HDR&);
		~HDR();

		bool Init(ID3D11Device1* device, int width, int height);
		void Shutdown();

	public:
		void Process(ID3D11DeviceContext1* mImmediateContext, std::unique_ptr<Canvas>const& Canvas);
		
	private:
		HRESULT OnResize(ID3D11Device1* device, int renderWidth, int renderHeight);

	private:
		std::unique_ptr<Texture> mLuminanceText;

		ID3D11ComputeShader* mLuminancePassCS;
		ID3D11ComputeShader* mToneMapPassCS;

		// TEMP
		ID3D11Buffer* mScreenQuadVB;
		ID3D11Buffer* mScreenQuadIB;

		ID3D11InputLayout* mDebugIL;
		ID3D11VertexShader* mDebugVS;
		ID3D11PixelShader* mDebugPS;

		struct MatrixBufferType
		{
			XMMATRIX gWorldProj;
		};

		ID3D11Buffer* MatrixBuffer;
		// save working space width
		int clientWidth;
		// save working space height
		int clientHeight;
	};
}

