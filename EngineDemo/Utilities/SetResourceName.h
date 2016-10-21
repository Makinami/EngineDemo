#pragma once

#include <d3d11_1.h>
#include <string>

template <typename T>
inline void SetDebugName(T &child, const std::string& name)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (child != nullptr && name != "")
		child->SetPrivateData(WKPDID_D3DDebugObjectName, name.size(), name.c_str());

#endif
}