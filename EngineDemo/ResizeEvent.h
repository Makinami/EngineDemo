#pragma once

#include <list>
#include <d3d11_1.h>

class OnResizeListener;
class OnResizeNotifier;

class OnResizeNotifier
{
public:
	OnResizeNotifier();
	~OnResizeNotifier();

	static HRESULT OnResize(ID3D11Device1* device, int renderWidth, int renderHeight);

private:
	friend class OnResizeListener;

	void RegisterListener(OnResizeListener* rh);
	void UnregisterListener(OnResizeListener* rh);

private:
	static OnResizeNotifier& Instance();

private:
	std::list<OnResizeListener*> ResizeNotifyTargets;
};

class OnResizeListener
{
protected:
	friend class OnResizeNotifier;

protected:
	OnResizeListener();
	virtual ~OnResizeListener();

	virtual HRESULT OnResize(ID3D11Device1* device, int renderWidth, int renderHeight) { return S_OK; };
};