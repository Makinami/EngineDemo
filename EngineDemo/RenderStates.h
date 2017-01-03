#pragma once

#include <d3d11_1.h>

class RenderStates
{
public:
	static HRESULT InitAll(ID3D11Device1* device);
	static void ReleaseAll();

	struct Sampler
	{
		static ID3D11SamplerState* BilinearClampSS;
		static ID3D11SamplerState* BilinearWrapSS;
		static ID3D11SamplerState* BilinearClampComLessSS;
		static ID3D11SamplerState* TrilinearClampSS;
		static ID3D11SamplerState* TrilinearWrapSS;
		static ID3D11SamplerState* AnisotropicClampSS;
		static ID3D11SamplerState* AnisotropicWrapSS;
	};

	struct DepthStencil
	{
		static ID3D11DepthStencilState* NoWriteGreaterEqualDSS;
		static ID3D11DepthStencilState* DefaultDSS;
		static ID3D11DepthStencilState* WriteNoTestDSS;
	};

	struct Rasterizer
	{
		static ID3D11RasterizerState1* DefaultRS;
		static ID3D11RasterizerState1* WireframeRS;
	};

	struct Blend
	{

	};

	RenderStates() = delete;
	RenderStates(const RenderStates&) = delete;
	RenderStates& operator=(const RenderStates&) = delete;
	RenderStates(RenderStates&&) = delete;
	RenderStates& operator=(RenderStates&&) = delete;
	~RenderStates() = delete;
};

