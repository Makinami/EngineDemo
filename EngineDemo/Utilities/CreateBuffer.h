#pragma once

#include "SetResourceName.h"

#include <d3d11_1.h>

template <typename T>
bool CreateConstantBuffer(ID3D11Device1* device, size_t size, T &cb, const std::string& name = "")
{
	D3D11_BUFFER_DESC constantBufferDesc;
	constantBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	constantBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	constantBufferDesc.MiscFlags = 0;
	constantBufferDesc.StructureByteStride = 0;
	constantBufferDesc.Usage = D3D11_USAGE_DYNAMIC;

	constantBufferDesc.ByteWidth = size;

	if (FAILED(device->CreateBuffer(&constantBufferDesc, nullptr, &cb))) return false;
	SetDebugName(cb, name);
	return true;
}