#pragma once

#include <d3d11_1.h>
#include <fstream>
#include <string>

#include <wrl\client.h>

#include "SetResourceName.h"

#include <filesystem>

#include <D3Dcompiler.h>
#pragma comment(lib, "D3DCompiler.lib")

namespace fs = std::experimental::filesystem;

// Load content of file given by 'fileName' to memory at 'data' alocating needed memory at returning size in 'size'. Calle is responsible for dealocating memory
bool LoadShader(_In_ std::wstring fileName, _Out_ char* &data, _Out_ size_t &size);

// Read compiled shader to the blob (only cso and hlsl in not deployment build)
void LoadCompiledShader(_In_ const fs::path& fileName, _Out_ ID3DBlob* &blob, _In_ const std::string target = "", _In_ const std::string entry_point = "main");

template <typename T>
bool CreatePSFromFile(_In_ const fs::path& fileName, _In_ ID3D11Device1 * device, _Out_opt_ T &ps, const std::string& name = "")
{
	ID3DBlob* blob{ nullptr };

	if (fileName.extension() != ".cso"
#ifndef DEPLOYMENT
		&& fileName.extension() != ".hlsl"
#endif
		)
		return false;

	LoadCompiledShader(fileName, blob, "ps_5_0");

	if (blob != nullptr && SUCCEEDED(device->CreatePixelShader(blob->GetBufferPointer(), blob->GetBufferSize(), 0, &ps)))
	{
		SetDebugName(ps, name);

		return true;
	}
	else
		return false;
}

template <typename T>
bool CreateCSFromFile(_In_ const fs::path& fileName, _In_ ID3D11Device1 * device, _Out_opt_ T &cs, const std::string& name = "")
{
	ID3DBlob* blob{ nullptr };

	if (fileName.extension() != ".cso"
#ifndef DEPLOYMENT
		&& fileName.extension() != ".hlsl"
#endif
		)
		return false;

	LoadCompiledShader(fileName, blob, "cs_5_0");

	if (blob != nullptr && SUCCEEDED(device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), 0, &cs)))
	{
		SetDebugName(cs, name);

		return true;
	}
	else
		return false;
}

template <typename T>
bool CreateDSFromFile(_In_ const fs::path& fileName, _In_ ID3D11Device1 * device, _Out_opt_ T &ds, const std::string& name = "")
{
	ID3DBlob* blob{ nullptr };

	if (fileName.extension() != ".cso"
#ifndef DEPLOYMENT
		&& fileName.extension() != ".hlsl"
#endif
		)
		return false;

	LoadCompiledShader(fileName, blob, "ds_5_0");

	if (blob != nullptr && SUCCEEDED(device->CreateDomainShader(blob->GetBufferPointer(), blob->GetBufferSize(), 0, &ds)))
	{
		SetDebugName(ds, name);

		return true;
	}
	else
		return false;
}

template <typename T>
bool CreateHSFromFile(_In_ const fs::path& fileName, _In_ ID3D11Device1 * device, _Out_opt_ T &hs, const std::string& name = "")
{
	ID3DBlob* blob{ nullptr };

	if (fileName.extension() != ".cso"
#ifndef DEPLOYMENT
		&& fileName.extension() != ".hlsl"
#endif
		)
		return false;

	LoadCompiledShader(fileName, blob, "hs_5_0");

	if (blob != nullptr && SUCCEEDED(device->CreateHullShader(blob->GetBufferPointer(), blob->GetBufferSize(), 0, &hs)))
	{
		SetDebugName(hs, name);

		return true;
	}
	else
		return false;
}

template <typename T>
bool CreateVSFromFile(_In_ const fs::path& fileName, _In_ ID3D11Device1 * device, _Out_opt_ T &vs, const std::string& name = "")
{
	ID3DBlob* blob{ nullptr };

	if (fileName.extension() != ".cso"
#ifndef DEPLOYMENT
		&& fileName.extension() != ".hlsl"
#endif
		)
		return false;

	LoadCompiledShader(fileName, blob, "vs_5_0");

	if (blob != nullptr && SUCCEEDED(device->CreateVertexShader(blob->GetBufferPointer(), blob->GetBufferSize(), 0, &vs)))
	{
		SetDebugName(vs, name);

		return true;
	}
	else
		return false;
}

template <typename T>
bool CreateGSFromFile(_In_ const fs::path& fileName, _In_ ID3D11Device1 * device, _Out_opt_ T &gs, const std::string& name = "")
{
	ID3DBlob* blob{ nullptr };

	if (fileName.extension() != ".cso"
#ifndef DEPLOYMENT
		&& fileName.extension() != ".hlsl"
#endif
		)
		return false;

	LoadCompiledShader(fileName, blob, "hs_5_0");

	if (blob != nullptr && SUCCEEDED(device->CreateGeometryShader(blob->GetBufferPointer(), blob->GetBufferSize(), 0, &gs)))
	{
		SetDebugName(gs, name);

		return true;
	}
	else
		return false;
}

template <typename Tvs, typename Til>
bool CreateVSAndInputLayout(_In_ const fs::path& fileName, _In_ ID3D11Device1 * device, _Out_opt_ Tvs &vs, const D3D11_INPUT_ELEMENT_DESC* ilDesc, const UINT numElements, _Out_opt_ Til &il, const std::string& vsname = "", const std::string& ilname = "")
{
	ID3DBlob* blob{ nullptr };

	if (fileName.extension() != ".cso"
#ifndef DEPLOYMENT
		&& fileName.extension() != ".hlsl"
#endif
		)
		return false;

	LoadCompiledShader(fileName, blob, "cs_5_0");

	if (blob != nullptr && SUCCEEDED(device->CreateVertexShader(blob->GetBufferPointer(), blob->GetBufferSize(), 0, &vs)) 
		&& SUCCEEDED(device->CreateInputLayout(ilDesc, numElements, blob->GetBufferPointer(), blob->GetBufferSize(), &il)))
	{
		SetDebugName(vs, vsname);
		SetDebugName(il, ilname);

		return true;
	}
	else
		return false;
}