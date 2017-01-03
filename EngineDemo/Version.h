#pragma once
#include <string>

namespace Version
{
	const unsigned int iMajor = 0;
	const unsigned int iMinor = 1;
	const unsigned int iPatch = 0;
	const unsigned int iBuild = 51;

	const std::wstring strMajor = std::to_wstring(iMajor);
	const std::wstring strMinor = std::to_wstring(iMinor);
	const std::wstring strPatch = std::to_wstring(iPatch);
	const std::wstring strBuild = std::to_wstring(iBuild);

	const std::wstring strVersion = strMajor + L"." + strMinor + L"." + strPatch;
	const std::wstring strVersionFull = strVersion + L"." + strBuild;
}
