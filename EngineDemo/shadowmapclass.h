#pragma once

/* temp */
//---------------------------------------------------------------------------------------
// Convenience macro for releasing COM objects.
//---------------------------------------------------------------------------------------

#define ReleaseCOM(x) { if(x){ x->Release(); x = nullptr; } }

//---------------------------------------------------------------------------------------
// Convenience macro for deleting objects.
//---------------------------------------------------------------------------------------

#define SafeDelete(x) { delete x; x = nullptr; }


#pragma comment(lib,  "d3d11.lib")

#include <d3d11_1.h>

class ShadowMapClass
{
public:
	ShadowMapClass(ID3D11Device1* device, UINT width, UINT height);
	~ShadowMapClass();

	ID3D11ShaderResourceView* DepthMapSRV();

	void BindDsvAndSetNullRenderTarget(ID3D11DeviceContext1* dc);
	void ClearDepthMap(ID3D11DeviceContext1* dc);

private:
	UINT mWidth;
	UINT mHeight;

	ID3D11ShaderResourceView* mDepthMapSRV;
	ID3D11DepthStencilView* mDepthMapDSV;

	D3D11_VIEWPORT mViewport;
};