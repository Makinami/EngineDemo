#include <cassert>

#include "ResizeEvent.h"

/*
* On resize notifier-listener pipeline
*/
OnResizeNotifier::OnResizeNotifier()
{
}

OnResizeNotifier::~OnResizeNotifier()
{
	assert(ResizeNotifyTargets.size() == 0);
}

HRESULT OnResizeNotifier::OnResize(ID3D11Device1 * device, int renderWidth, int renderHeight)
{
	HRESULT hr;
	for (auto&& it : OnResizeNotifier::Instance().ResizeNotifyTargets)
		hr = it->OnResize(device, renderWidth, renderHeight);
	return S_OK;
}

void OnResizeNotifier::RegisterListener(OnResizeListener * rh)
{
	ResizeNotifyTargets.push_back(rh);
}

void OnResizeNotifier::UnregisterListener(OnResizeListener * rh)
{
	ResizeNotifyTargets.remove(rh);
}

OnResizeNotifier & OnResizeNotifier::Instance()
{
	static OnResizeNotifier me;
	return me;
}

OnResizeListener::OnResizeListener()
{
	OnResizeNotifier::Instance().RegisterListener(this);
}

OnResizeListener::~OnResizeListener()
{
	OnResizeNotifier::Instance().UnregisterListener(this);
}