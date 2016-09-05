#pragma once

#include <d3d11_1.h>

#define ReleaseCOM(x) { if(x){ x->Release(); x = nullptr; } }
#define SafeDelete(x) { delete x; x = nullptr; }

#pragma warning(disable:4838)
#include <DirectXMath.h>
#pragma warning(default:4838

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

	class HDR
	{
	public:
		HDR();
		HDR(const HDR&);
		~HDR();

		bool Init(ID3D11Device1* device, int width, int height);
		void Shutdown();

	public:
		void StarRegister(ID3D11DeviceContext1* mImmediateContext);
		void StopRegister(ID3D11DeviceContext1* mImmediateContext);

		void Process(ID3D11DeviceContext1* mImmediateContext);
		void Present(ID3D11DeviceContext1* mImmediateContext);

		void SetRenderTarget(ID3D11DeviceContext1* mImmediateContext);
		void ClearRenderTarget(ID3D11DeviceContext1* mImmediateContext, DirectX::XMFLOAT4& colour);
		ID3D11ShaderResourceView* GetShaderResourceView() const;

	private:
		std::unique_ptr<Texture> mHDRText;
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

