#pragma once

#include <d3d11_1.h>

template <typename BufType>
void MapResources(ID3D11DeviceContext1* mImmediateContext, ID3D11Buffer* constantBuffer, BufType &resources, size_t size = 0, D3D11_MAP MapFlags = D3D11_MAP_WRITE_DISCARD)
{
	D3D11_MAPPED_SUBRESOURCE mappedResources;
	mImmediateContext->Map(constantBuffer, 0, MapFlags, 0, &mappedResources);
	memcpy(mappedResources.pData, &resources, size ? size : sizeof(BufType));
	mImmediateContext->Unmap(constantBuffer, 0);
}