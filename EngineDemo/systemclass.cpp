#include "systemclass.h"

namespace
{
	SystemClass* callbackSystem = nullptr;
}

LRESULT CALLBACK MainWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	return callbackSystem->MsgProc(hwnd, msg, wParam, lParam);
}


SystemClass::SystemClass(HINSTANCE hInstance)
: mhAppInstance(hInstance),
  mhMainWnd(0),
  mAppName(L"EngineDemo"),
  mWndCap(L"EngineDemo"),
  mClientHeight(720),
  mClientWidth(1280)
{}

SystemClass::SystemClass(const SystemClass &)
{}

SystemClass::~SystemClass()
{
	ShutdownMainWindow();

	mhAppInstance = NULL;

	callbackSystem = nullptr;
}

bool SystemClass::Init(std::string filename)
{
	Settings = std::make_unique<INIReader>(filename);

	if (Settings->ParseError() < 0) return false;

	mClientHeight = Settings->GetInteger("Window", "height", mClientHeight);
	mClientWidth = Settings->GetInteger("Window", "width", mClientWidth);

	if (!InitMainWindow()) return false;

	return true;
}

void SystemClass::Shutdown()
{
	ShutdownMainWindow();
}

int SystemClass::Run()
{
	MSG msg = {0};

	while (msg.message != WM_QUIT)
	{
		if (PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			TranslateMessage( &msg );
			DispatchMessage( &msg ); 
		}
		else
		{
			Sleep(100);
		}
	}

	return (int)msg.wParam;
}

LRESULT SystemClass::MsgProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	return DefWindowProc(hwnd, msg, wParam, lParam);
}

bool SystemClass::Frame()
{
	return false;
}

bool SystemClass::InitMainWindow()
{
	WNDCLASS wc = {0};
	wc.style = CS_HREDRAW|CS_VREDRAW;
	wc.lpfnWndProc = MainWndProc;
	wc.hInstance = mhAppInstance;
	wc.hIcon = LoadCursor(NULL, IDI_WINLOGO);
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)GetStockObject(NULL_BRUSH);
	wc.lpszClassName = mAppName;

	if (!RegisterClass(&wc))
	{
		MessageBox(0, L"RegisterClass Failed.", 0, 0);
		return false;
	}

	// Compute window rectangle dimensions based on requested client area dimensions.
	RECT R = { 0, 0, mClientWidth, mClientHeight };
	AdjustWindowRect(&R, WS_OVERLAPPEDWINDOW, false);
	auto width = R.right - R.left;
	auto height = R.bottom - R.top;

	mhMainWnd = CreateWindow(mAppName, mWndCap.c_str(),
		WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, width, height, 0, 0, mhAppInstance, 0);
	if (!mhMainWnd)
	{
		MessageBox(0, L"CreateWindow Failes.", 0, 0);
		return false;
	}

	ShowWindow(mhMainWnd, SW_SHOW);
	UpdateWindow(mhMainWnd);

	return true;
}

void SystemClass::ShutdownMainWindow()
{
	DestroyWindow(mhMainWnd);
	mhMainWnd = NULL;

	UnregisterClass(mAppName, mhAppInstance);
}
