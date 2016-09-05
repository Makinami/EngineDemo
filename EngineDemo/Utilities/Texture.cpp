#include "Texture.h"

#include <algorithm>

using namespace std;
//using namespace DirectX;

ID3D11Device1* TextureFactory::device = nullptr;

void TextureFactory::SetDevice(ID3D11Device1 * _device)
{
	device = _device;
}

std::unique_ptr<Texture> TextureFactory::CreateTexture(UINT bind, DXGI_FORMAT format, UINT width, int height, int depth)
{
	if (!device) return nullptr;
	
	if (depth > 1)
	{
		D3D11_TEXTURE3D_DESC textDesc;

		textDesc.Width = width;
		textDesc.Height = height;
		textDesc.Depth = depth;
		textDesc.MipLevels = 1;
		textDesc.Format = format;
		textDesc.Usage = D3D11_USAGE_DEFAULT;
		textDesc.BindFlags = bind;
		textDesc.CPUAccessFlags = 0;
		textDesc.MiscFlags = 0;

		return CreateTexture(textDesc);
	}
	else if (height > 1 || depth < -1)
	{
		D3D11_TEXTURE2D_DESC textDesc;

		textDesc.Width = width;
		textDesc.Height = height;
		textDesc.MipLevels = 1;
		textDesc.ArraySize = (depth < 1 ? -depth : 1);
		textDesc.Format = format;
		textDesc.SampleDesc = { 1, 0 };
		textDesc.Usage = D3D11_USAGE_DEFAULT;
		textDesc.BindFlags = bind;
		textDesc.CPUAccessFlags = 0;
		textDesc.MiscFlags = 0;

		return CreateTexture(textDesc);
	}
	else
	{
		return nullptr;
	}
}

std::unique_ptr<Texture> TextureFactory::CreateTexture(const D3D11_TEXTURE2D_DESC & textDesc)
{
	if (!device) return nullptr;

	unique_ptr<Texture> texture = make_unique<Texture>();

	texture->TextureType = texture->Texture2D;
	texture->bind = textDesc.BindFlags;

	if (FAILED(device->CreateTexture2D(&textDesc, nullptr, &texture->mTexture2D)))
		return nullptr;

	// Create RTV
	if (textDesc.BindFlags & D3D11_BIND_RENDER_TARGET)
	{
		D3D11_RENDER_TARGET_VIEW_DESC rtvDesc;

		rtvDesc.Format = textDesc.Format;
		if (textDesc.ArraySize > 1)
		{
			rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
			rtvDesc.Texture2DArray.MipSlice = 0;
			rtvDesc.Texture2DArray.FirstArraySlice = 0;
			rtvDesc.Texture2DArray.ArraySize = textDesc.ArraySize;
		}
		else
		{
			rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
			rtvDesc.Texture2D.MipSlice = 0;
		}

		if (FAILED(device->CreateRenderTargetView(texture->mTexture2D.Get(), &rtvDesc, &texture->mRenderTargetView)))
			return nullptr;
	}

	// Create DSV
	if (textDesc.BindFlags & D3D11_BIND_DEPTH_STENCIL)
	{
		D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc;

		dsvDesc.Format = textDesc.Format;
		if (textDesc.ArraySize > 1)
		{
			dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
			dsvDesc.Texture2DArray.MipSlice = 0;
			dsvDesc.Texture2DArray.FirstArraySlice = 0;
			dsvDesc.Texture2DArray.ArraySize = textDesc.ArraySize;
		}
		else
		{
			dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
			dsvDesc.Texture2D.MipSlice = 0;
		}

		if (FAILED(device->CreateDepthStencilView(texture->mTexture2D.Get(), &dsvDesc, &texture->mDepthStencilView)))
			return nullptr;
	}

	// Create SRV
	if (textDesc.BindFlags & D3D11_BIND_SHADER_RESOURCE)
	{
		D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;

		srvDesc.Format = textDesc.Format;
		if (textDesc.ArraySize > 1)
		{
			srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
			srvDesc.Texture2DArray.MostDetailedMip = 0;
			srvDesc.Texture2DArray.MipLevels = -1;
			srvDesc.Texture2DArray.FirstArraySlice = 0;
			srvDesc.Texture2DArray.ArraySize = textDesc.ArraySize;
		}
		else
		{
			srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
			srvDesc.Texture2D.MostDetailedMip = 0;
			srvDesc.Texture2D.MipLevels = -1;
		}

		if (FAILED(device->CreateShaderResourceView(texture->mTexture2D.Get(), &srvDesc, &texture->mShaderResourceView)))
			return nullptr;
	}

	// Create UAV
	if (textDesc.BindFlags & D3D11_BIND_UNORDERED_ACCESS)
	{
		D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc;

		uavDesc.Format = textDesc.Format;
		if (textDesc.ArraySize > 1)
		{
			uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2DARRAY;
			uavDesc.Texture2DArray.MipSlice = 0;
			uavDesc.Texture2DArray.FirstArraySlice = 0;
			uavDesc.Texture2DArray.ArraySize = textDesc.ArraySize;
		}
		else
		{
			uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
			uavDesc.Texture2D.MipSlice = 0;
		}

		if (FAILED(device->CreateUnorderedAccessView(texture->mTexture2D.Get(), &uavDesc, &texture->mUnorderedAccessView)))
			return nullptr;
	}

	return texture;
}

std::unique_ptr<Texture> TextureFactory::CreateTexture(const D3D11_TEXTURE3D_DESC & textDesc)
{
	if (!device) return nullptr;

	unique_ptr<Texture> texture = make_unique<Texture>();

	texture->TextureType = texture->Texture3D;
	texture->bind = textDesc.BindFlags;

	if (FAILED(device->CreateTexture3D(&textDesc, nullptr, &texture->mTexture3D)))
		return nullptr;

	// Create RTV
	if (textDesc.BindFlags & D3D11_BIND_RENDER_TARGET)
	{
		D3D11_RENDER_TARGET_VIEW_DESC rtvDesc;

		rtvDesc.Format = textDesc.Format;
		rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE3D;
		rtvDesc.Texture3D.MipSlice = 0;
		rtvDesc.Texture3D.FirstWSlice = 0;
		rtvDesc.Texture3D.WSize = -1;

		if (FAILED(device->CreateRenderTargetView(texture->mTexture3D.Get(), &rtvDesc, &texture->mRenderTargetView)))
			return nullptr;
	}
	
	// Create SRV
	if (textDesc.BindFlags & D3D11_BIND_SHADER_RESOURCE)
	{
		D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;

		srvDesc.Format = textDesc.Format;
		srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE3D;
		srvDesc.Texture3D.MostDetailedMip = 0;
		srvDesc.Texture3D.MipLevels = -1;

		if (FAILED(device->CreateShaderResourceView(texture->mTexture3D.Get(), &srvDesc, &texture->mShaderResourceView)))
			return nullptr;
	}

	// Create UAV
	if (textDesc.BindFlags & D3D11_BIND_UNORDERED_ACCESS)
	{
		D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc;

		uavDesc.Format = textDesc.Format;
		uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE3D;
		uavDesc.Texture3D.MipSlice = 0;
		uavDesc.Texture3D.FirstWSlice = 0;
		uavDesc.Texture3D.WSize = -1;

		if (FAILED(device->CreateUnorderedAccessView(texture->mTexture3D.Get(), &uavDesc, &texture->mUnorderedAccessView)))
			return nullptr;
	}

	return texture;
}


/*
 	Texture Definition
*/
Texture::Texture() :
	bind(0),
	TextureType(TextureXD),
	mTexture1D(nullptr)
{
	ZeroMemory(&textDesc1D, max({ sizeof(textDesc1D), sizeof(textDesc2D), sizeof(textDesc3D) }));
}

ID3D11RenderTargetView * Texture::GetRTV() const
{
	return mRenderTargetView.Get();
}

ID3D11DepthStencilView * Texture::GetDSV() const
{
	return mDepthStencilView.Get();
}

ID3D11ShaderResourceView * Texture::GetSRV() const
{
	return mShaderResourceView.Get();
}

ID3D11UnorderedAccessView * Texture::GetUAV() const
{
	return mUnorderedAccessView.Get();
}

ID3D11RenderTargetView * const* Texture::GetAddressOfRTV() const
{
	return mRenderTargetView.GetAddressOf();
}

ID3D11DepthStencilView * const* Texture::GetAddressOfDSV() const
{
	return mDepthStencilView.GetAddressOf();
}

ID3D11ShaderResourceView * const* Texture::GetAddressOfSRV() const
{
	return mShaderResourceView.GetAddressOf();
}

ID3D11UnorderedAccessView * const* Texture::GetAddressOfUAV() const
{
	return mUnorderedAccessView.GetAddressOf();
}

ID3D11Resource * Texture::GetTexture() const
{
	switch (TextureType)
	{
		case Texture1D: 
			return mTexture1D.Get();

		case Texture2D: 
			return mTexture2D.Get();

		case Texture3D: 
			return mTexture3D.Get();

		case TextureXD:
		default:
			return nullptr;
	}
}
// End Texture Definition
