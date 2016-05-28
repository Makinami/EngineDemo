#include "CreateShader.h"

bool LoadShader(_In_ std::wstring fileName, _Out_ char* &data, _Out_ size_t &size)
{
	std::ifstream stream;

	stream.open(fileName.c_str(), std::ifstream::binary);
	if (stream.good())
	{
		stream.seekg(0, std::ios::end);
		size = size_t(stream.tellg());
		data = new char[size];
		stream.seekg(0, std::ios::beg);
		stream.read(data, size);
		stream.close();

		return true;
	}

	return false;
}

#ifdef NOT
bool CreatePSFromFile(_In_ std::wstring fileName, ID3D11Device1 * device, ID3D11PixelShader* &ps)
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

		delete[] data;
		return true;
	}

	return false;
}

bool CreateCSFromFile(_In_ std::wstring fileName, ID3D11Device1 * device, ID3D11ComputeShader* &cs)
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

		delete[] data;
		return true;
	}

	return false;
}

bool CreateDSFromFile(_In_ std::wstring fileName, ID3D11Device1 * device, ID3D11DomainShader* &ds)
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

		delete[] data;
		return true;
	}

	return false;
}

bool CreateHSFromFile(_In_ std::wstring fileName, ID3D11Device1 * device, ID3D11HullShader* &hs)
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

		delete[] data;
		return true;
	}

	return false;
}

bool CreateVSFromFile(_In_ std::wstring fileName, ID3D11Device1 * device, ID3D11VertexShader* &vs)
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

		delete[] data;
		return true;
	}

	return false;
}

bool CreateGSFromFile(_In_ std::wstring fileName, ID3D11Device1 * device, ID3D11GeometryShader* &gs)
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

		delete[] data;
		return true;
	}

	return false;
}

bool CreateVSAndInputLayout(_In_ std::wstring fileName, ID3D11Device1 * device, ID3D11VertexShader* &vs, const D3D11_INPUT_ELEMENT_DESC* ilDesc, UINT numElements, ID3D11InputLayout* &il)
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
			vs->Release();
			vs = nullptr;

			delete[] data;
			return false;
		}

		delete[] data;
		return true;
	}

	return false;
}
#endif