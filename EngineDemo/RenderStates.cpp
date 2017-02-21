#include "RenderStates.h"

#include <float.h>

#define ReleaseCOM(x) { if(x){ x->Release(); x = nullptr; } }
#define SafeDelete(x) { delete x; x = nullptr; }

#define EXIT_ON_FAILURE(fnc)  \
	{ \
		HRESULT result; \
		if (FAILED(result = fnc)) { \
			return result; \
		} \
	}

ID3D11SamplerState*		 RenderStates::Sampler::BilinearClampSS				= nullptr;
ID3D11SamplerState*		 RenderStates::Sampler::BilinearWrapSS				= nullptr;
ID3D11SamplerState*		 RenderStates::Sampler::BilinearClampComLessSS		= nullptr;
ID3D11SamplerState*		 RenderStates::Sampler::TrilinearClampSS			= nullptr;
ID3D11SamplerState*		 RenderStates::Sampler::TrilinearWrapSS				= nullptr;
ID3D11SamplerState*		 RenderStates::Sampler::AnisotropicClampSS			= nullptr;
ID3D11SamplerState*		 RenderStates::Sampler::AnisotropicWrapSS			= nullptr;

ID3D11DepthStencilState* RenderStates::DepthStencil::NoWriteGreaterEqualDSS	= nullptr;
ID3D11DepthStencilState* RenderStates::DepthStencil::DefaultDSS				= nullptr;
ID3D11DepthStencilState* RenderStates::DepthStencil::WriteNoTestDSS			= nullptr;

ID3D11RasterizerState1*  RenderStates::Rasterizer::DefaultRS				= nullptr;
ID3D11RasterizerState1*	 RenderStates::Rasterizer::NoCullingRS				= nullptr;
ID3D11RasterizerState1*  RenderStates::Rasterizer::WireframeRS				= nullptr;

HRESULT RenderStates::InitAll(ID3D11Device1 * device)
{
	// Bilinear Clamp
	D3D11_SAMPLER_DESC BilinearClampDesc = {};
	BilinearClampDesc.Filter = D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT;
	BilinearClampDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
	BilinearClampDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
	BilinearClampDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
	BilinearClampDesc.MinLOD = -FLT_MAX;
	BilinearClampDesc.MaxLOD = FLT_MAX;
	BilinearClampDesc.MipLODBias = 0.0f;
	BilinearClampDesc.MaxAnisotropy = 1;
	BilinearClampDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
	BilinearClampDesc.BorderColor[0] =
		BilinearClampDesc.BorderColor[1] =
		BilinearClampDesc.BorderColor[2] =
		BilinearClampDesc.BorderColor[3] = 0.0f;
	EXIT_ON_FAILURE(device->CreateSamplerState(&BilinearClampDesc, &Sampler::BilinearClampSS));

	// Bilinear Wrap
	D3D11_SAMPLER_DESC BilinearWrapDesc = {};
	BilinearWrapDesc.Filter = D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT;
	BilinearWrapDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	BilinearWrapDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	BilinearWrapDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
	BilinearWrapDesc.MinLOD = -FLT_MAX;
	BilinearWrapDesc.MaxLOD = FLT_MAX;
	BilinearWrapDesc.MipLODBias = 0.0f;
	BilinearWrapDesc.MaxAnisotropy = 1;
	BilinearWrapDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
	BilinearWrapDesc.BorderColor[0] =
		BilinearWrapDesc.BorderColor[1] =
		BilinearWrapDesc.BorderColor[2] =
		BilinearWrapDesc.BorderColor[3] = 0.0f;
	EXIT_ON_FAILURE(device->CreateSamplerState(&BilinearWrapDesc, &Sampler::BilinearWrapSS));

	// Bilinear Clamp Com Less
	D3D11_SAMPLER_DESC BilinearClampComLessDesc = {};
	BilinearClampComLessDesc.Filter = D3D11_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
	BilinearClampComLessDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
	BilinearClampComLessDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
	BilinearClampComLessDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
	BilinearClampComLessDesc.MinLOD = -FLT_MAX;
	BilinearClampComLessDesc.MaxLOD = FLT_MAX;
	BilinearClampComLessDesc.MipLODBias = 0.0f;
	BilinearClampComLessDesc.MaxAnisotropy = 1;
	BilinearClampComLessDesc.ComparisonFunc = D3D11_COMPARISON_LESS;
	BilinearClampComLessDesc.BorderColor[0] =
		BilinearClampComLessDesc.BorderColor[1] =
		BilinearClampComLessDesc.BorderColor[2] =
		BilinearClampComLessDesc.BorderColor[3] = 0.0f;
	EXIT_ON_FAILURE(device->CreateSamplerState(&BilinearClampComLessDesc, &Sampler::BilinearClampComLessSS));

	// Trilinear Clamp
	D3D11_SAMPLER_DESC TrilinearClampDesc = {};
	TrilinearClampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	TrilinearClampDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
	TrilinearClampDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
	TrilinearClampDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
	TrilinearClampDesc.MinLOD = -FLT_MAX;
	TrilinearClampDesc.MaxLOD = FLT_MAX;
	TrilinearClampDesc.MipLODBias = 0.0f;
	TrilinearClampDesc.MaxAnisotropy = 1;
	TrilinearClampDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
	TrilinearClampDesc.BorderColor[0] =
		TrilinearClampDesc.BorderColor[1] =
		TrilinearClampDesc.BorderColor[2] =
		TrilinearClampDesc.BorderColor[3] = 0.0f;
	EXIT_ON_FAILURE(device->CreateSamplerState(&TrilinearClampDesc, &Sampler::TrilinearClampSS));

	// Trilinear Wrap
	D3D11_SAMPLER_DESC TrilinearWrapDesc = {};
	TrilinearWrapDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	TrilinearWrapDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	TrilinearWrapDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	TrilinearWrapDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
	TrilinearWrapDesc.MinLOD = -FLT_MAX;
	TrilinearWrapDesc.MaxLOD = FLT_MAX;
	TrilinearWrapDesc.MipLODBias = 0.0f;
	TrilinearWrapDesc.MaxAnisotropy = 1;
	TrilinearWrapDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
	TrilinearWrapDesc.BorderColor[0] =
		TrilinearWrapDesc.BorderColor[1] =
		TrilinearWrapDesc.BorderColor[2] =
		TrilinearWrapDesc.BorderColor[3] = 0.0f;
	EXIT_ON_FAILURE(device->CreateSamplerState(&TrilinearWrapDesc, &Sampler::TrilinearWrapSS));

	// Anisotropic Clamp
	D3D11_SAMPLER_DESC AnisotropicClampDesc = {};
	AnisotropicClampDesc.Filter = D3D11_FILTER_ANISOTROPIC;
	AnisotropicClampDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
	AnisotropicClampDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
	AnisotropicClampDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
	AnisotropicClampDesc.MinLOD = -FLT_MAX;
	AnisotropicClampDesc.MaxLOD = FLT_MAX;
	AnisotropicClampDesc.MipLODBias = 0.0f;
	AnisotropicClampDesc.MaxAnisotropy = 16;
	AnisotropicClampDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
	AnisotropicClampDesc.BorderColor[0] =
		AnisotropicClampDesc.BorderColor[1] =
		AnisotropicClampDesc.BorderColor[2] =
		AnisotropicClampDesc.BorderColor[3] = 0.0f;
	EXIT_ON_FAILURE(device->CreateSamplerState(&AnisotropicClampDesc, &Sampler::AnisotropicClampSS));

	// Anisotropic Wrap
	D3D11_SAMPLER_DESC AnisotropicWrapDesc = {};
	AnisotropicWrapDesc.Filter = D3D11_FILTER_ANISOTROPIC;
	AnisotropicWrapDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	AnisotropicWrapDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	AnisotropicWrapDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
	AnisotropicWrapDesc.MinLOD = -FLT_MAX;
	AnisotropicWrapDesc.MaxLOD = FLT_MAX;
	AnisotropicWrapDesc.MipLODBias = 0.0f;
	AnisotropicWrapDesc.MaxAnisotropy = 16;
	AnisotropicWrapDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
	AnisotropicWrapDesc.BorderColor[0] =
		AnisotropicWrapDesc.BorderColor[1] =
		AnisotropicWrapDesc.BorderColor[2] =
		AnisotropicWrapDesc.BorderColor[3] = 0.0f;
	EXIT_ON_FAILURE(device->CreateSamplerState(&AnisotropicWrapDesc, &Sampler::AnisotropicWrapSS));

	// Default depth - reverse-Z
	// Depth - enable; Write - yes; Com - >; Sentil -disable
	D3D11_DEPTH_STENCIL_DESC DefaultDSDesc;
	DefaultDSDesc.DepthEnable = true;
	DefaultDSDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	DefaultDSDesc.DepthFunc = D3D11_COMPARISON_GREATER;
	DefaultDSDesc.StencilEnable = false;
	EXIT_ON_FAILURE(device->CreateDepthStencilState(&DefaultDSDesc, &DepthStencil::DefaultDSS));

	// Depth - enable; Write - no; Com - <=; Stencil - disable
	D3D11_DEPTH_STENCIL_DESC NoWriteGreaterEqualDesc;
	NoWriteGreaterEqualDesc.DepthEnable = true;
	NoWriteGreaterEqualDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
	NoWriteGreaterEqualDesc.DepthFunc = D3D11_COMPARISON_GREATER_EQUAL;
	NoWriteGreaterEqualDesc.StencilEnable = false;
	EXIT_ON_FAILURE(device->CreateDepthStencilState(&NoWriteGreaterEqualDesc, &DepthStencil::NoWriteGreaterEqualDSS));

	// Depth - enable; Write - enable; Test - none; Stencil - disable
	D3D11_DEPTH_STENCIL_DESC WriteNoTestDesc;
	WriteNoTestDesc.DepthEnable = true;
	WriteNoTestDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
	WriteNoTestDesc.DepthFunc = D3D11_COMPARISON_ALWAYS;
	WriteNoTestDesc.StencilEnable = false;
	EXIT_ON_FAILURE(device->CreateDepthStencilState(&WriteNoTestDesc, &DepthStencil::WriteNoTestDSS));

	// Rasterizer - default
	D3D11_RASTERIZER_DESC1 DefaulRasterizerDesc = {};
	DefaulRasterizerDesc.FillMode = D3D11_FILL_SOLID;
	DefaulRasterizerDesc.CullMode = D3D11_CULL_BACK;
	DefaulRasterizerDesc.FrontCounterClockwise = false;
	DefaulRasterizerDesc.DepthClipEnable = true;
	EXIT_ON_FAILURE(device->CreateRasterizerState1(&DefaulRasterizerDesc, &Rasterizer::DefaultRS));

	// Rasterizer - no culling
	D3D11_RASTERIZER_DESC1 NoCullingRasterizerDesc = {};
	NoCullingRasterizerDesc.FillMode = D3D11_FILL_SOLID;
	NoCullingRasterizerDesc.CullMode = D3D11_CULL_NONE;
	NoCullingRasterizerDesc.FrontCounterClockwise = false;
	NoCullingRasterizerDesc.DepthClipEnable = true;
	EXIT_ON_FAILURE(device->CreateRasterizerState1(&NoCullingRasterizerDesc, &Rasterizer::NoCullingRS));

	// Rasterizer - wireframe
	D3D11_RASTERIZER_DESC1 WireframeDesc = {};
	WireframeDesc.FillMode = D3D11_FILL_WIREFRAME;
	WireframeDesc.CullMode = D3D11_CULL_BACK;
	WireframeDesc.FrontCounterClockwise = false;
	WireframeDesc.DepthClipEnable = true;
	EXIT_ON_FAILURE(device->CreateRasterizerState1(&WireframeDesc, &Rasterizer::WireframeRS));

	return S_OK;
}

void RenderStates::ReleaseAll()
{
	ReleaseCOM(Sampler::BilinearClampSS);
	ReleaseCOM(Sampler::BilinearWrapSS);
	ReleaseCOM(Sampler::BilinearClampComLessSS);
	ReleaseCOM(Sampler::TrilinearClampSS);
	ReleaseCOM(Sampler::TrilinearWrapSS);
	ReleaseCOM(Sampler::AnisotropicClampSS);
	ReleaseCOM(Sampler::AnisotropicWrapSS);

	ReleaseCOM(DepthStencil::NoWriteGreaterEqualDSS);
	ReleaseCOM(DepthStencil::DefaultDSS);
	ReleaseCOM(DepthStencil::WriteNoTestDSS);

	ReleaseCOM(Rasterizer::DefaultRS);
	ReleaseCOM(Rasterizer::WireframeRS);
}
