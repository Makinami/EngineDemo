#pragma once

#include <d3d11_1.h>
#include <string>

inline void SetDebugName(ID3D11DeviceChild* child, const std::string& name)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (child != nullptr && name != "")
		child->SetPrivateData(WKPDID_D3DDebugObjectName, name.size(), name.c_str());

#endif
}