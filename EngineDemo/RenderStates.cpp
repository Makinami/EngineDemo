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

ID3D11SamplerState*		 RenderStates::Sampler::TriLinearClampSS		 = nullptr;
ID3D11SamplerState*		 RenderStates::Sampler::TriLinearWrapSS			 = nullptr;
ID3D11SamplerState*		 RenderStates::Sampler::AnisotropicWrapSS		 = nullptr;

ID3D11DepthStencilState* RenderStates::DepthStencil::NoWriteGreaterEqualDSS	= nullptr;
ID3D11DepthStencilState* RenderStates::DepthStencil::DefaultDSS				= nullptr;
ID3D11DepthStencilState* RenderStates::DepthStencil::WriteNoTestDSS			= nullptr;

ID3D11RasterizerState1*  RenderStates::Rasterizer::DefaultRS			 = nullptr;
ID3D11RasterizerState1*  RenderStates::Rasterizer::WireframeRS			 = nullptr;

HRESULT RenderStates::InitAll(ID3D11Device1 * device)
{
	// Trilinear Clamp
	D3D11_SAMPLER_DESC TriLinearClampDesc = {};
	TriLinearClampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	TriLinearClampDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
	TriLinearClampDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
	TriLinearClampDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
	TriLinearClampDesc.MinLOD = -FLT_MAX;
	TriLinearClampDesc.MaxLOD = FLT_MAX;
	TriLinearClampDesc.MipLODBias = 0.0f;
	TriLinearClampDesc.MaxAnisotropy = 1;
	TriLinearClampDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
	TriLinearClampDesc.BorderColor[0] =
		TriLinearClampDesc.BorderColor[1] =
		TriLinearClampDesc.BorderColor[2] =
		TriLinearClampDesc.BorderColor[3] = 0.0f;
	EXIT_ON_FAILURE(device->CreateSamplerState(&TriLinearClampDesc, &Sampler::TriLinearClampSS));

	// Trilinear Wrap
	D3D11_SAMPLER_DESC TriLinearWrapDesc = {};
	TriLinearWrapDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	TriLinearWrapDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	TriLinearWrapDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	TriLinearWrapDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
	TriLinearWrapDesc.MinLOD = -FLT_MAX;
	TriLinearWrapDesc.MaxLOD = FLT_MAX;
	TriLinearWrapDesc.MipLODBias = 0.0f;
	TriLinearWrapDesc.MaxAnisotropy = 1;
	TriLinearWrapDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
	TriLinearWrapDesc.BorderColor[0] =
		TriLinearWrapDesc.BorderColor[1] =
		TriLinearWrapDesc.BorderColor[2] =
		TriLinearWrapDesc.BorderColor[3] = 0.0f;
	EXIT_ON_FAILURE(device->CreateSamplerState(&TriLinearWrapDesc, &Sampler::TriLinearWrapSS));

	// Trilinear Wrap
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
	WriteNoTestDesc.DepthEnable = false;
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
	ReleaseCOM(Sampler::TriLinearClampSS);
	ReleaseCOM(Sampler::TriLinearWrapSS);
	ReleaseCOM(Sampler::AnisotropicWrapSS);

	ReleaseCOM(DepthStencil::NoWriteGreaterEqualDSS);

	ReleaseCOM(Rasterizer::DefaultRS);
	ReleaseCOM(Rasterizer::WireframeRS);
}
