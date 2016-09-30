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

namespace PostFX
{
	struct VertexType
	{
		XMFLOAT2 pos;
		XMFLOAT2 tex;
	};

	class Canvas
	{
	public:
		bool Init(ID3D11Device1* device, int width, int height);
		void Shutdown();

	public:
		void StartRegister(ID3D11DeviceContext1* mImmediateContext) const;
		void StopRegister(ID3D11DeviceContext1* mImmediateContext) const;

		void Swap();

		void Present(ID3D11DeviceContext1*& mImmediateContext);

	public:
		ID3D11ShaderResourceView*const* GetAddressOfSRV(bool secondary = false) const;
		ID3D11UnorderedAccessView*const* GetAddressOfUAV(bool secondary = false) const;

	private:
		std::unique_ptr<Texture> mMain;
		std::unique_ptr<Texture> mSecondary;

		std::unique_ptr<Texture> mDepthStencil;
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

	class HDR
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
	};
}
