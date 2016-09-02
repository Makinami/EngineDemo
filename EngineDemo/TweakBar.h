#pragma once

#pragma comment(lib, "AntTweakBar/lib/AntTweakBar.lib")

#include <AntTweakBar.h>
#include <d3d11_1.h>
#include <memory>

class TweakBar
{
public:
	TweakBar();
	~TweakBar();
public:
	HRESULT Init(ID3D11Device1* device, ID3D11DeviceContext1* mImmediateContext);

	int Draw();

public:
	TwBar* bar;
};

