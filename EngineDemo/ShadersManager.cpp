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