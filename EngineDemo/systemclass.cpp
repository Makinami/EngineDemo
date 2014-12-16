#include "systemclass.h"

namespace
{
	SystemClass* callbackSystem = nullptr;
}

LRESULT CALLBACK MainWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	return callbackSystem->MsgProc(hwnd, msg, wParam, lParam);
}

LRESULT CALLBACK StatusWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	return callbackSystem->StatusWndMsgProc(hwnd, msg, wParam, lParam);
}


SystemClass::SystemClass(HINSTANCE hInstance)
: mhAppInstance(hInstance),
  mhMainWnd(0),
  mAppName(L"EngineDemo"),
  mWndCap(L"EngineDemo"),
  mClientHeight(720),
  mClientWidth(1280)
{
	callbackSystem = this;
}

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
	Settings = std::make_shared<INIReader>(filename);

	Logger = std::make_shared<LoggerClass>(Settings->GetWString("General", "logfile", L"log.txt"), (HWND)0);

	if (!CreateStatusWindow())
		Logger->Notice(L"Could not create status window. Prociding without it.");
	else if (Logger->SetWindow(mEdit))
		Logger->Success(L"Created and assigned status window.");

	if (Settings->ParseError() < 0) Logger->Notice(L"Could not open file. Will use default settings.");
	else if (Settings->ParseError() > 0) Logger->Notice(L"Could not parse one or more lines (first line: " + std::to_wstring(Settings->ParseError()) + L"). Will skip them and use defaults instead.");
	else Logger->Success(L"Setting loaded.");

	mClientHeight = Settings->GetInteger("Window", "height", mClientHeight);
	mClientWidth = Settings->GetInteger("Window", "width", mClientWidth);

	if (!InitMainWindow()) return false;
	Logger->Success(L"Main window created.");

	if (mD3D = std::make_shared<D3DClass>())
	{
		mD3D->SetLogger(Logger);
		if (!mD3D->Init(mhMainWnd, mClientWidth, mClientHeight, Settings)) return false;
	}
	Logger->Success(L"DirectX initiated.");
	
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
			mD3D->BeginScene();
			mD3D->EndScene();
			Sleep(100);
		}
	}

	return (int)msg.wParam;
}

LRESULT SystemClass::MsgProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	switch (msg)
	{
		case WM_KEYDOWN:
			if (wParam == VK_F11)
			{
				if (IsWindowVisible(mStatusWnd)) ShowWindow(mStatusWnd, SW_HIDE);
				else ShowWindow(mStatusWnd, SW_SHOWNA);
			}
			return 0;
		case WM_DESTROY:
			PostQuitMessage(0);
			return 0;
	}

	return DefWindowProc(hwnd, msg, wParam, lParam);
}

LRESULT SystemClass::StatusWndMsgProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	int wmId, wmEvent;
	PAINTSTRUCT ps;
	HDC hdc;
	int mClientWidth;
	int mClientHeight;

	switch (msg)
	{
		case WM_SIZE:
			mClientWidth = LOWORD(lParam);
			mClientHeight = HIWORD(lParam);

			SetWindowPos(mEdit, 0, 0, 0, mClientWidth, mClientHeight, SWP_NOMOVE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_SHOWWINDOW);
			return DefWindowProc(hwnd, msg, wParam, lParam);
		case WM_KEYDOWN:
			if (wParam == VK_F11) ShowWindow(mStatusWnd, SW_HIDE);
			return 0;
		case WM_CLOSE:
			ShowWindow(mStatusWnd, SW_HIDE);
			return 0;
		default:
			return DefWindowProc(hwnd, msg, wParam, lParam);
	}
	return 0;
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
		Logger->Error(L"Main window class not registered. Exiting...");
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
		Logger->Error(L"Main window not created. Exiting...");
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

bool SystemClass::CreateStatusWindow()
{
	WNDCLASS wc = { 0 };
	wc.style = CS_HREDRAW | CS_VREDRAW;
	wc.lpfnWndProc = StatusWndProc;
	wc.hInstance = mhAppInstance;
	wc.hIcon = LoadCursor(NULL, IDI_WINLOGO);
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)GetStockObject(NULL_BRUSH);
	wc.lpszClassName = L"EngineDemoStatusWnd";

	if (!RegisterClass(&wc)) return false;

	mStatusWnd = CreateWindow(L"EngineDemoStatusWnd", NULL, WS_VISIBLE | WS_OVERLAPPED, CW_USEDEFAULT, 0, 600, 300, NULL, NULL, mhAppInstance, NULL);

	if (!mStatusWnd)
	{
		return FALSE;
	}

	RECT R;
	GetClientRect(mStatusWnd, &R);

	mEdit = CreateWindow(WC_EDIT, L"    Status window initiated.", WS_VSCROLL | ES_READONLY | WS_HSCROLL | WS_VISIBLE | WS_CHILD | ES_MULTILINE, R.left, R.top, R.right, R.bottom, mStatusWnd, NULL, mhAppInstance, NULL);

	ShowWindow(mStatusWnd, SW_SHOW);
	UpdateWindow(mStatusWnd);

	return true;
}
