#pragma once

#include <d3d11_1.h>
#include <fstream>
#include <string>

#include <wrl\client.h>

#include "SetResourceName.h"

bool LoadShader(_In_ std::wstring fileName, _Out_ char* &data, _Out_ size_t &size);

template <typename T>
bool CreatePSFromFile(_In_ std::wstring fileName, ID3D11Device1 * device, T &ps, const std::string& name = "")
{
	size_t size;
	char* data;

	if (LoadShader(fileName, data, size))
	{
		if (FAILED(device->CreatePixelShader(data, size, 0, &ps)))
		{
			delete[] data;
			return false;
		}

		SetDebugName(ps, name);

		delete[] data;
		return true;
	}

	return false;
}

//template <typename T> bool CreateCSFromFile(_In_ std::wstring fileName, ID3D11Device1 * device, T &cs);
template <typename T>
bool CreateCSFromFile(_In_ std::wstring fileName, ID3D11Device1 * device, T &cs, const std::string& name = "")
{
	size_t size;
	char* data;

	if (LoadShader(fileName, data, size))
	{
		if (FAILED(device->CreateComputeShader(data, size, 0, &cs)))
		{
			delete[] data;
			return false;
		}

		SetDebugName(cs, name);

		delete[] data;
		return true;
	}

	return false;
}

template <typename T>
bool CreateDSFromFile(_In_ std::wstring fileName, ID3D11Device1 * device, T &ds, const std::string& name = "")
{
	size_t size;
	char* data;

	if (LoadShader(fileName, data, size))
	{
		if (FAILED(device->CreateDomainShader(data, size, 0, &ds)))
		{
			delete[] data;
			return false;
		}

		SetDebugName(ds, name);

		delete[] data;
		return true;
	}

	return false;
}

template <typename T>
bool CreateHSFromFile(_In_ std::wstring fileName, ID3D11Device1 * device, T &hs, const std::string& name = "")
{
	size_t size;
	char* data;

	if (LoadShader(fileName, data, size))
	{
		if (FAILED(device->CreateHullShader(data, size, 0, &hs)))
		{
			delete[] data;
			return false;
		}

		SetDebugName(hs, name);

		delete[] data;
		return true;
	}

	return false;
}

template <typename T>
bool CreateVSFromFile(_In_ std::wstring fileName, ID3D11Device1 * device, T &vs, const std::string& name = "")
{
	size_t size;
	char* data;

	if (LoadShader(fileName, data, size))
	{
		if (FAILED(device->CreateVertexShader(data, size, 0, &vs)))
		{
			delete[] data;
			return false;
		}

		SetDebugName(vs, name);

		delete[] data;
		return true;
	}

	return false;
}

template <typename T>
bool CreateGSFromFile(_In_ std::wstring fileName, ID3D11Device1 * device, T &gs, const std::string& name = "")
{
	size_t size;
	char* data;

	if (LoadShader(fileName, data, size))
	{
		if (FAILED(device->CreateGeometryShader(data, size, 0, &gs)))
		{
			delete[] data;
			return false;
		}

		SetDebugName(gs, name);

		delete[] data;
		return true;
	}

	return false;
}

template <typename Tvs, typename Til>
bool CreateVSAndInputLayout(_In_ std::wstring fileName, ID3D11Device1 * device, Tvs &vs, const D3D11_INPUT_ELEMENT_DESC* ilDesc, UINT numElements, Til &il, const std::string& vsname = "", const std::string& ilname = "")
{
	size_t size;
	char* data;

	if (LoadShader(fileName, data, size))
	{
		if (FAILED(device->CreateVertexShader(data, size, 0, &vs)))
		{
			delete[] data;
			return false;
		}

		if (FAILED(device->CreateInputLayout(ilDesc, numElements, data, size, &il)))
		{
			//vs->Release();
			vs = nullptr;

			delete[] data;
			return false;
		}

		SetDebugName(vs, vsname);
		SetDebugName(il, ilname);

		delete[] data;
		return true;
	}

	return false;
}