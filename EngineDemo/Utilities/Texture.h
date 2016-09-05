#pragma once

#define NOMINMAX

// DirectX
#include <d3d11_1.h>
// ComPtr
#include <wrl\client.h>
// unique_ptr
#include <memory>

// Helpers
#define ReleaseCOM(x) { if(x){ x->Release(); x = nullptr; } }
#define SafeDelete(x) { delete x; x = nullptr; }

#define EXIT_ON_FAILURE(fnc)  \
	{ \
		HRESULT result; \
		if (FAILED(result = (fnc))) { \
			return result; \
		} \
	}

#define EXIT_ON_NULL(fnc)  \
	{ \
		if ((fnc) == nullptr) { \
			return E_FAIL; \
		} \
	}

// Classes' declaration
class Texture;
class TextureFactory;

class TextureFactory
{
private:
	TextureFactory() {};

	TextureFactory(TextureFactory const &) = delete;
	void operator=(TextureFactory const &) = delete;

public:
	static void SetDevice(ID3D11Device1* _device);
	static std::unique_ptr<Texture> CreateTexture(UINT bind, DXGI_FORMAT format, UINT width, int height = 1, int depth = 1);
	static std::unique_ptr<Texture> CreateTexture(const D3D11_TEXTURE2D_DESC& textDesc);
	static std::unique_ptr<Texture> CreateTexture(const D3D11_TEXTURE3D_DESC& textDesc);
private:
	static ID3D11Device1* device;
};

class Texture
{
	friend TextureFactory;
public:
	Texture();
	Texture(Texture const &) = delete;
	void operator=(Texture const &) = delete;
	~Texture() {};

	ID3D11RenderTargetView *    GetRTV() const;
	ID3D11DepthStencilView *	GetDSV() const;
	ID3D11ShaderResourceView *  GetSRV() const;
	ID3D11UnorderedAccessView * GetUAV() const;

	ID3D11RenderTargetView* const*	  GetAddressOfRTV() const;
	ID3D11DepthStencilView* const*	  GetAddressOfDSV() const;
	ID3D11ShaderResourceView* const*  GetAddressOfSRV() const;
	ID3D11UnorderedAccessView* const* GetAddressOfUAV() const;

	ID3D11Resource * GetTexture() const;

private:
	UINT bind;

	enum {
		TextureXD, Texture1D, Texture2D, Texture3D
	} TextureType;

	union {
		D3D11_TEXTURE1D_DESC textDesc1D;
		D3D11_TEXTURE2D_DESC textDesc2D;
		D3D11_TEXTURE3D_DESC textDesc3D;
	};

	union {
		Microsoft::WRL::ComPtr<ID3D11Texture1D> mTexture1D;
		Microsoft::WRL::ComPtr<ID3D11Texture2D> mTexture2D;
		Microsoft::WRL::ComPtr<ID3D11Texture3D> mTexture3D;
	};

	Microsoft::WRL::ComPtr<ID3D11RenderTargetView> mRenderTargetView;
	Microsoft::WRL::ComPtr<ID3D11DepthStencilView> mDepthStencilView;
	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> mShaderResourceView;
	Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> mUnorderedAccessView;
};