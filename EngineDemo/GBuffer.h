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

class GBufferClass
{
	public:
		GBufferClass();
		~GBufferClass() {
			Shutdown();
		};

		int Init(ID3D11Device1* device, int width, int height);
		void Shutdown();

		void SetBufferRTV(ID3D11DeviceContext1* mImmediateContext) const;
		void UnsetBufferRTV(ID3D11DeviceContext1* mImmediateContext) const;
		
		void SetBufferSRV(ID3D11DeviceContext1* mImmediateContext, int first_slot = 0);
		void UnsetBufferSRV(ID3D11DeviceContext1* mImmediateContext) const;

	private:
		static const int buffer_count = 2;
		int srv_slot = -1;

		std::unique_ptr<Texture> mGBuffer[buffer_count];

		ID3D11RenderTargetView* mGBufferRTV[buffer_count];
		ID3D11ShaderResourceView* mGBufferSRV[buffer_count];

		std::unique_ptr<Texture> mDepthStencil;
		D3D11_VIEWPORT mViewPort;
};

