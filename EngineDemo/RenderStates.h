#pragma once

#include <d3d11_1.h>

class RenderStates
{
public:
	static HRESULT InitAll(ID3D11Device1* device);
	static void ReleaseAll();

	struct Sampler
	{
		static ID3D11SamplerState* TriLinearClampSS;
		static ID3D11SamplerState* TriLinearWrapSS;
		static ID3D11SamplerState* AnisotropicWrapSS;
	};

	struct DepthStencil
	{
		static ID3D11DepthStencilState* NoWriteLessEqualDSS;
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

