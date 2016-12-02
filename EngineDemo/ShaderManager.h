#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <filesystem>

#include <d3d11_1.h>

class ShaderManager
{
	using fspath = std::tr2::sys::path;

public:
	static std::shared_ptr<ShaderManager> Instance();

	void ReleaseAll();

	ID3D11PixelShader* GetPS(std::string identifier);
	//ID3D11VertexShader* GetVS(std::string identifier);
	ID3D11HullShader* GetHS(std::string identifier);
	ID3D11DomainShader* GetDS(std::string identifier);
	ID3D11GeometryShader* GetGS(std::string identifier);
	ID3D11ComputeShader* GetCS(std::string identifier);

public:
	ShaderManager(ShaderManager const&) = delete;
	void operator=(ShaderManager const&) = delete;
	~ShaderManager();

	void SetDevice(ID3D11Device1* const& _device) { device = _device; };

protected:
	ShaderManager() {};

private:
	ID3D11PixelShader* AddPS(std::string& identifier);
	//ID3D11VertexShader* AddVS(std::string identifier);
	ID3D11HullShader* AddHS(std::string identifier);
	ID3D11DomainShader* AddDS(std::string identifier);
	ID3D11GeometryShader* AddGS(std::string identifier);
	ID3D11ComputeShader* AddCS(std::string identifier);

	fspath ResolveIdentifierToPath(std::string& identifier);

private:
	static std::shared_ptr<ShaderManager> mInstance;

	ID3D11Device1* device;

	fspath mBaseDir;
	fspath mShaderDir;
	
	std::unordered_map<std::string, ID3D11PixelShader*> PS;
	//std::unordered_map<std::string, ID3D11VertexShader*> VS;
	std::unordered_map<std::string, ID3D11HullShader*> HS;
	std::unordered_map<std::string, ID3D11DomainShader*> DS;
	std::unordered_map<std::string, ID3D11GeometryShader*> GS;
	std::unordered_map<std::string, ID3D11ComputeShader*> CS;
};