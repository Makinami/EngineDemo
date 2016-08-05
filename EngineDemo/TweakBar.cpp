#include "TweakBar.h"



TweakBar::TweakBar()
{
}


TweakBar::~TweakBar()
{
	TwTerminate();
}

HRESULT TweakBar::Init(ID3D11Device1 * device, ID3D11DeviceContext1 * mImmediateContext)
{
	TwInit(TW_DIRECT3D11, device);
	TwWindowSize(1280, 720);

	bar = TwNewBar("NameOfMyTweakBar");

	return S_OK;
}

int TweakBar::Draw()
{
	return TwDraw();
}
