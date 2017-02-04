#include "ShadersManager.h"

#include "Utilities\CreateShader.h"

#include <algorithm>
#include <regex>

#define ReleaseCOM(x) { if(x){ x->Release(); x = nullptr; } }

using namespace std;
using namespace std::tr2::sys;

std::shared_ptr<ShadersManager> ShadersManager::mInstance = nullptr;

std::shared_ptr<ShadersManager> ShadersManager::Instance()
{
	struct OpenShaderManager : public ShadersManager {};
	return mInstance ? mInstance : mInstance = make_shared<OpenShaderManager>();
}

void ShadersManager::ReleaseAll()
{
	std::for_each(begin(PS), end(PS), [](auto&& it) { 
		ReleaseCOM(it.second); 
	});
	PS.clear();
}

ID3D11PixelShader * ShadersManager::GetPS(string identifier)
{
	auto shader = PS.find(identifier);
	return shader != end(PS) ? shader->second : AddPS(identifier);
}

ID3D11HullShader * ShadersManager::GetHS(std::string identifier)
{
	auto shader = HS.find(identifier);
	return shader != end(HS) ? shader->second : AddHS(identifier);
}

ID3D11DomainShader * ShadersManager::GetDS(std::string identifier)
{
	auto shader = DS.find(identifier);
	return shader != end(DS) ? shader->second : AddDS(identifier);
}

ID3D11GeometryShader * ShadersManager::GetGS(std::string identifier)
{
	auto shader = GS.find(identifier);
	return shader != end(GS) ? shader->second : AddGS(identifier);
}

ID3D11ComputeShader * ShadersManager::GetCS(std::string identifier)
{
	auto shader = CS.find(identifier);
	return shader != end(CS) ? shader->second : AddCS(identifier);
}

ShadersManager::~ShadersManager()
{
	ReleaseAll();
}

ID3D11PixelShader * ShadersManager::AddPS(string& identifier)
{
	fspath shaderfile = ResolveIdentifierToPath(identifier);
	
	ID3D11PixelShader* shader;
	CreatePSFromFile(shaderfile, device, shader);

	return PS[identifier] = shader;
}

ID3D11HullShader * ShadersManager::AddHS(std::string identifier)
{
	fspath shaderfile = ResolveIdentifierToPath(identifier);

	ID3D11HullShader* shader;
	CreateHSFromFile(shaderfile, device, shader);

	return HS[identifier] = shader;
}

ID3D11DomainShader * ShadersManager::AddDS(std::string identifier)
{
	fspath shaderfile = ResolveIdentifierToPath(identifier);

	ID3D11DomainShader* shader;
	CreateDSFromFile(shaderfile, device, shader);

	return DS[identifier] = shader;
}

ID3D11GeometryShader * ShadersManager::AddGS(std::string identifier)
{
	fspath shaderfile = ResolveIdentifierToPath(identifier);

	ID3D11GeometryShader* shader;
	CreateGSFromFile(shaderfile, device, shader);

	return GS[identifier] = shader;
}

ID3D11ComputeShader * ShadersManager::AddCS(std::string identifier)
{
	fspath shaderfile = ResolveIdentifierToPath(identifier);

	ID3D11ComputeShader* shader;
	CreateCSFromFile(shaderfile, device, shader);

	return CS[identifier] = shader;
}

path ShadersManager::ResolveIdentifierToPath(std::string & identifier)
{
	if (!(mShaderDir != path()))
	{
		fspath pathDif = "../Debug";
		mBaseDir = current_path() / pathDif;
		mShaderDir = mBaseDir / "Shaders";
	}

	return (mShaderDir / std::regex_replace(identifier, std::regex("::"), "/")).replace_extension(".cso");
}

inline std::ostream& operator<< (std::ostream& stream, ShaderTypes type)
{
	switch (type)
	{
	case ShaderTypes::Compute:
		return stream << "Compute Shader";
	case ShaderTypes::Pixel:
		return stream << "Pixel Shader";
	case ShaderTypes::Vertex:
		return stream << "Vertex Shader";
	case ShaderTypes::Domain:
		return stream << "Domain Shader";
	case ShaderTypes::Hull:
		return stream << "Hull Shader";
	default:
		return stream;
	}
}

ShaderMap FindShaderFiles(const fs::path& directory)
{
	ShaderMap shaders;

	const regex shader_pattern("(.*)([VHDPC]S)([0-9._]*)$");
	smatch results;

	for (auto& file : fs::directory_iterator(directory))
	{
		if (fs::is_regular_file(file) && (file.path().extension() == ".cso" || file.path().extension() == ".hlsl"))
		{
			auto str = file.path().stem().string();
			if (regex_search(str, results, shader_pattern))
			{
				shaders[ShaderFile::StrToEnum(results[2].str())].emplace_back(ShaderFile::StrToEnum(results[2].str()), results[1].str(), results[3].str(), file.path());
			}
		}
	}

	return shaders;
}

namespace ImGui
{
	static auto shader_file_vector_getter = [](void* vec, int idx, const char** out_text)
	{
		auto& vector = *static_cast<std::vector<ShaderFile>*>(vec);
		if (idx < 0 || idx >= vector.size()) { return false; }
		*out_text = vector.at(idx).readable_id.c_str();
		return true;
	};

	bool Combo(const char* label, int* currIndex, std::vector<ShaderFile>& values)
	{
		if (values.empty()) return false;
		return Combo(label, currIndex, shader_file_vector_getter,
			static_cast<void*>(&values), values.size());
	}
}

// String must be a valid representetion of shader type (PS,etc.). Otherwise throws std::logic_error exception.
ShaderTypes ShaderFile::StrToEnum(const std::string& _type)
{
	if (_type == "PS")
		return ShaderTypes::Pixel;
	else if (_type == "VS")
		return ShaderTypes::Vertex;
	else if (_type == "DS")
		return ShaderTypes::Domain;
	else if (_type == "HS")
		return ShaderTypes::Hull;
	else if (_type == "CS")
		return ShaderTypes::Compute;
	else
		throw std::logic_error("No shader of type " + _type);
}

bool ShaderFile::CheckType(const std::string& _type)
{
	if (_type == "PS" || _type == "VS" || _type == "DS" || _type == "CS" || _type == "HS")
		return true;
	else
		return false;
}

ShaderFile::ShaderFile(const ShaderTypes& _type, const std::string& _name, const std::string& _version, const fspath& _file) noexcept
	: type(_type), name(_name), version(_version), file(_file), readable_id(name + " " + version)
{ }

// Type string must be a valid representetion of shader type (PS,etc.). Otherwise throws std::logic_error exception.
ShaderFile::ShaderFile(const std::string& _type, const std::string& _name, const std::string& _version, const fspath _file)
	: type(StrToEnum(_type)), name(_name), version(_version), file(_file), readable_id(name + " " + version)
{ }

std::ostream& operator<< (std::ostream& stream, const ShaderFile& shader)
{
	stream << "Name: " << shader.name << "\tType: " << shader.type << "\tVer.: " << shader.version << endl;
	return stream;
}