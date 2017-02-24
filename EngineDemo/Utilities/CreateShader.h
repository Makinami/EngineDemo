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
[[deprecated("Use LoadCompileShader insteead")]]
bool LoadShader(_In_ std::wstring fileName, _Out_ char* &data, _Out_ size_t &size);

// Read compiled shader to the blob (only cso and hlsl in not deployment build)
void LoadCompiledShader(_In_ const fs::path& fileName, _Out_ Microsoft::WRL::ComPtr<ID3DBlob> &blob, _In_ const std::string& target = "", _In_ const std::string& entry_point = "main");

template <typename T>
bool CreatePSFromFile(_In_ const fs::path& fileName, _In_ ID3D11Device1 * &device, _Out_opt_ T &ps, const std::string& name = "")
{
	Microsoft::WRL::ComPtr<ID3DBlob> shaderBlob{ nullptr };
	ID3D11PixelShader* shader{ nullptr };

	LoadCompiledShader(fileName, shaderBlob, "ps_5_0");

	if (shaderBlob != nullptr && SUCCEEDED(device->CreatePixelShader(shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize(), 0, &shader)))
	{
		SetDebugName(shader, name);

		ps = std::move(shader);

		return true;
	}
	else
		return false;
}

template <typename T>
bool CreateCSFromFile(_In_ const fs::path& fileName, _In_ ID3D11Device1 * &device, _Out_opt_ T &cs, const std::string& name = "")
{
	Microsoft::WRL::ComPtr<ID3DBlob> shaderBlob{ nullptr };
	ID3D11ComputeShader* shader{ nullptr };

	LoadCompiledShader(fileName, shaderBlob, "cs_5_0");

	if (shaderBlob != nullptr && SUCCEEDED(device->CreateComputeShader(shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize(), 0, &shader)))
	{
		SetDebugName(shader, name);

		cs = std::move(shader);

		return true;
	}
	else
		return false;
}

template <typename T>
bool CreateDSFromFile(_In_ const fs::path& fileName, _In_ ID3D11Device1 * &device, _Out_opt_ T &ds, const std::string& name = "")
{
	Microsoft::WRL::ComPtr<ID3DBlob> shaderBlob{ nullptr };
	ID3D11DomainShader* shader{ nullptr };

	LoadCompiledShader(fileName, shaderBlob, "ds_5_0");

	if (shaderBlob != nullptr && SUCCEEDED(device->CreateDomainShader(shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize(), 0, &shader)))
	{
		SetDebugName(shader, name);

		ds = std::move(shader);

		return true;
	}
	else
		return false;
}

template <typename T>
bool CreateHSFromFile(_In_ const fs::path& fileName, _In_ ID3D11Device1 * &device, _Out_opt_ T &hs, const std::string& name = "")
{
	Microsoft::WRL::ComPtr<ID3DBlob> shaderBlob{ nullptr };
	ID3D11HullShader* shader{ nullptr };

	LoadCompiledShader(fileName, shaderBlob, "hs_5_0");

	if (shaderBlob != nullptr && SUCCEEDED(device->CreateHullShader(shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize(), 0, &shader)))
	{
		SetDebugName(shader, name);

		hs = std::move(shader);

		return true;
	}
	else
		return false;
}

template <typename T>
bool CreateVSFromFile(_In_ const fs::path& fileName, _In_ ID3D11Device1 * &device, _Out_opt_ T &vs, const std::string& name = "")
{
	Microsoft::WRL::ComPtr<ID3DBlob> blob{ nullptr };

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
	Microsoft::WRL::ComPtr<ID3DBlob> shaderBlob{ nullptr };
	ID3D11GeometryShader* shader{ nullptr };

	LoadCompiledShader(fileName, shaderBlob, "hs_5_0");

	if (shaderBlob != nullptr && SUCCEEDED(device->CreateGeometryShader(shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize(), 0, &shader)))
	{
		SetDebugName(shader, name);

		gs = std::move(shader);

		return true;
	}
	else
		return false;
}

template <typename Tvs, typename Til>
bool CreateVSAndInputLayout(_In_ const fs::path& fileName, _In_ ID3D11Device1 * device, _Out_opt_ Tvs &vs, const D3D11_INPUT_ELEMENT_DESC* ilDesc, const UINT numElements, _Out_opt_ Til &il, const std::string& vsname = "", const std::string& ilname = "")
{
	Microsoft::WRL::ComPtr<ID3DBlob> shaderBlob{ nullptr };
	ID3D11VertexShader* shader{ nullptr };
	ID3D11InputLayout* input{ nullptr };

	LoadCompiledShader(fileName, shaderBlob, "vs_5_0");

	if (shaderBlob != nullptr && SUCCEEDED(device->CreateVertexShader(shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize(), 0, &shader)) 
		&& SUCCEEDED(device->CreateInputLayout(ilDesc, numElements, shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize(), &input)))
	{
		SetDebugName(shader, vsname);
		SetDebugName(input, ilname);

		vs = shader;
		il = input;

		return true;
	}
	else
		return false;
}