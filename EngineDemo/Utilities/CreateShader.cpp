#include "CreateShader.h"

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

void LoadCompiledShader(const fs::path & fileName, ID3DBlob *& blob, const std::string target, const std::string entry_point)
{
#ifndef DEPLOYMENT
	if (fileName.extension() == ".hlsl")
	{
		D3DCompileFromFile(fileName.wstring().c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, entry_point.c_str(), target.c_str(),
			D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION | D3DCOMPILE_ENABLE_BACKWARDS_COMPATIBILITY, 0, &blob, nullptr);
	}
	else
#endif
	{
		D3DReadFileToBlob(fileName.wstring().c_str(), &blob);
	}
}