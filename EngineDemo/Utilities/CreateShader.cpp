#include "CreateShader.h"

#include <string>
#include <codecvt>

#include "..\loggerclass.h"

// Load content of file given by 'fileName' to memory at 'data' alocating needed memory at returning size in 'size'.
// Calle is responsible for dealocating memory.
bool LoadShader(std::wstring fileName, char *& data, size_t & size)
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

void LoadCompiledShader(const fs::path & fileName, ID3DBlob *& shaderBlob, const std::string target, const std::string entry_point)
{
	HRESULT hr;
	ID3DBlob* errorBlob{ nullptr };
#ifndef DEPLOYMENT
	if (fileName.extension() == ".hlsl")
	{
		hr = D3DCompileFromFile(fileName.wstring().c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, entry_point.c_str(), target.c_str(),
			D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION | D3DCOMPILE_ENABLE_BACKWARDS_COMPATIBILITY, 0, &shaderBlob, &errorBlob);
	}
#endif
	if (fileName.extension() == ".cso")
	{
		hr = D3DReadFileToBlob(fileName.wstring().c_str(), &shaderBlob);
	}

	if (shaderBlob == nullptr)
	{
		if (errorBlob != nullptr)
		{
			std::wstring_convert<std::codecvt_utf8<wchar_t>> myconv;
			Logger::Instance().Error(L"Failed to compile shader \"" + fileName.wstring() + L"\":\n" 
				+ myconv.from_bytes(std::string((char*)errorBlob->GetBufferPointer(), 
									errorBlob->GetBufferSize()-2))); // don't include '\n'
		}
		else
		{
			switch (hr)
			{
				case D3D11_ERROR_FILE_NOT_FOUND:
					Logger::Instance().Error(L"Shader file \"" + fileName.wstring() + L"\" not found");
					break;
				default:
					Logger::Instance().Error(L"Failed to compile shader \"" + fileName.wstring() + L"\": reason unknown ");
					break;
			}
		}
	}
}