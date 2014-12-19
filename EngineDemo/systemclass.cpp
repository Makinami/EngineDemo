#include "systemclass.h"

// Hack for inability to set class member function as window proc
namespace
{
	SystemClass* callbackSystem = nullptr;
}

// Main/game window proc
LRESULT CALLBACK MainWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	return callbackSystem->MsgProc(hwnd, msg, wParam, lParam);
}

// Status window proc
LRESULT CALLBACK StatusWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	return callbackSystem->StatusWndMsgProc(hwnd, msg, wParam, lParam);
}


// ------------------------------------------------------------------------
//                           SystemClass definition
// ------------------------------------------------------------------------

// Set basic values
SystemClass::SystemClass(HINSTANCE hInstance)
: mhAppInstance(hInstance),
  mhMainWnd(0),
  mAppName(L"EngineDemo"),
  mWndCap(L"EngineDemo"),
  mClientHeight(720),
  mClientWidth(1280),
  mAppPaused(false),
  mWndState(WndStateNormal)
{
	callbackSystem = this;
}

SystemClass::SystemClass(const SystemClass &)
{}

SystemClass::~SystemClass()
{
	// Shutdown system
	Shutdown();

	callbackSystem = nullptr;
}

// Init all systems
bool SystemClass::Init(std::string filename)
{
	// Load setting
	Settings = std::make_shared<INIReader>(filename);

	// Create logger
	Logger = std::make_shared<LoggerClass>(Settings->GetWString("General", "logfile", L"log.txt"), (HWND)0);

	if (!CreateStatusWindow())
		Logger->Notice(L"Could not create status window. Prociding without it.");
	else if (Logger->SetWindow(mEdit))
		Logger->Success(L"Created and assigned status window.");

	if (Settings->ParseError() < 0) Logger->Notice(L"Could not open file. Will use default settings.");
	else if (Settings->ParseError() > 0) Logger->Notice(L"Could not parse one or more lines (first line: " + std::to_wstring(Settings->ParseError()) + L"). Will skip them and use defaults instead.");
	else Logger->Success(L"Setting loaded.");

	// Set client size
	mClientHeight = Settings->GetInteger("Window", "height", mClientHeight);
	mClientWidth = Settings->GetInteger("Window", "width", mClientWidth);

	// Init main window
	if (!InitMainWindow()) return false;
	Logger->Success(L"Main window created.");

	// Create D3D class
	D3D = std::make_shared<D3DClass>();
	
	// Give D3D logger and initiate
	D3D->SetLogger(Logger);
	if (!D3D->Init(mhMainWnd, mClientWidth, mClientHeight, Settings)) return false;
	Logger->Success(L"DirectX initiated.");

	Input = std::make_shared<InputClass>();
	
	if (!Input->Init(mhAppInstance, mhMainWnd, mClientWidth, mClientHeight))
	{
		Logger->Error(L"Failed to initiate input");
		return false;
	}
	Logger->Success(L"Input initiated");

	Camera = std::make_shared<CameraClass>();

	Timer = std::make_shared<TimerClass>();

	// temp
	mLootAtPosition = XMFLOAT3(0, 0, 0);
	mPhi = XM_PIDIV4;
	mTheta = XM_PIDIV2;
	mRadius = 5;

	return true;
}

// Shut down everything
void SystemClass::Shutdown()
{
	D3D->Shutdown();

	ShutdownMainWindow();
}

// Main loop (for now just outline)
int SystemClass::Run()
{
	Timer->Reset();

	MSG msg = { 0 };

	while (msg.message != WM_QUIT)
	{
		if (PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		else
		{
			if ( !mAppPaused )
			{
				Timer->Tick();
				if (!Frame()) SendMessage(mhMainWnd, WM_QUIT, 0, 0);
			}
			else
			{
				Sleep(100);
			}
			
		}
	}

	return (int)msg.wParam;
}

// Main/game window's proc
LRESULT SystemClass::MsgProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	switch (msg)
	{
		case WM_KEYDOWN:
			// Show/hide status window
			if (wParam == VK_F11)
			{
				if (IsWindowVisible(mStatusWnd)) ShowWindow(mStatusWnd, SW_HIDE);
				else ShowWindow(mStatusWnd, SW_SHOWNA);
			}
			return 0;
		case WM_SIZE:
			mClientWidth = LOWORD(lParam);
			mClientHeight = HIWORD(lParam);
			if ( D3D )
			{
				if ( wParam == SIZE_MINIMIZED )
				{
					mAppPaused = true;
					mWndState = WndStateMinimized;
					// timer
				}
				else if ( wParam == SIZE_MAXIMIZED )
				{
					mAppPaused = false;
					mWndState = WndStateMaximized;
					// timer
				}
				else if ( wParam == SIZE_RESTORED )
				{
					// Restoring from minimized
					if ( mWndState == WndStateMinimized )
					{
						mAppPaused = false;
						mWndState = WndStateNormal;
						D3D->OnResize(mClientWidth, mClientHeight);
						//timer
					}
					// Restoring from maximized
					else if ( mWndState == WndStateMaximized )
					{
						mAppPaused = false;
						mWndState = WndStateNormal;
						D3D->OnResize(mClientWidth, mClientHeight);
					}
					else if ( mWndState == WndStateResizing )
					{
						// Dropedl; served at the end in WM_EXITMOVE
					}
					else
					{
						D3D->OnResize(mClientWidth, mClientHeight);
					}
				}
			}
			break;
		case WM_ENTERSIZEMOVE:
			mAppPaused = true;
			mWndState = WndStateResizing;
			// timer
			return 0;
		case WM_EXITSIZEMOVE:
			mAppPaused = false;
			mWndState = WndStateNormal;
			// timer
			D3D->OnResize(mClientWidth, mClientHeight);
			return 0;
		
		// Limit window size
		case WM_GETMINMAXINFO:
			((MINMAXINFO*)lParam)->ptMinTrackSize.x = 600;
			((MINMAXINFO*)lParam)->ptMinTrackSize.y = 400;
			return 0;
		case WM_LBUTTONDOWN:
		case WM_MBUTTONDOWN:
		case WM_RBUTTONDOWN:
			OnMouseDown(wParam, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
			return 0;
		case WM_LBUTTONUP:
		case WM_MBUTTONUP:
		case WM_RBUTTONUP:
			OnMouseUp(wParam, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
			return 0;
		case WM_MOUSEMOVE:
			OnMouseMove(wParam, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
			return 0;
		case WM_DESTROY:
			PostQuitMessage(0);
			return 0;
	}

	return DefWindowProc(hwnd, msg, wParam, lParam);
}

LRESULT SystemClass::StatusWndMsgProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	int mStatusWidth;
	int mStatusHeight;
	switch (msg)
	{
		case WM_SIZE:
			// Keep edit control the same size as status window
			mStatusWidth = LOWORD(lParam);
			mStatusHeight = HIWORD(lParam);

			SetWindowPos(mEdit, 0, 0, 0, mStatusWidth, mStatusHeight, SWP_NOMOVE | SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_SHOWWINDOW);
			return DefWindowProc(hwnd, msg, wParam, lParam);
		case WM_KEYDOWN:
			// Hide status window
			if (wParam == VK_F11) ShowWindow(mStatusWnd, SW_HIDE);
			return 0;
		case WM_CLOSE:
			// Instead of closing status window, hide it
			ShowWindow(mStatusWnd, SW_HIDE);
			return 0;
		default:
			return DefWindowProc(hwnd, msg, wParam, lParam);
	}
	return 0;
}

// Run every frame
bool SystemClass::Frame()
{
	float dt = Timer->DeltaTime();
	Input->Capture();

	// Convert Spherical to Cartesian coordinates.
	float x = mRadius*sinf(mPhi)*cosf(mTheta);
	float z = mRadius*sinf(mPhi)*sinf(mTheta);
	float y = mRadius*cosf(mPhi);

	if (GetAsyncKeyState('W') & 0x8000) mLootAtPosition.y += 5*dt;
	if (GetAsyncKeyState('S') & 0x8000) mLootAtPosition.y -= 5*dt;
	if (GetAsyncKeyState('A') & 0x8000) mLootAtPosition.x -= 5*dt;
	if (GetAsyncKeyState('D') & 0x8000) mLootAtPosition.x += 5*dt;

	XMVECTOR pos = XMVectorSet(x+mLootAtPosition.x, y+mLootAtPosition.y, z+mLootAtPosition.z, 1.0f);
	XMVECTOR target = XMVectorSet(mLootAtPosition.x, mLootAtPosition.y, mLootAtPosition.z, 0.0f);
	XMVECTOR up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);

	D3D->SetViewMatrix(XMMatrixLookAtLH(pos, target, up));

	D3D->BeginScene();
	D3D->Render();
	D3D->EndScene();

	return true;
}

// Create main/game window
bool SystemClass::InitMainWindow()
{
	WNDCLASS wc = { 0 };
	wc.style = CS_HREDRAW | CS_VREDRAW;
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

// Close main/game window
void SystemClass::ShutdownMainWindow()
{
	DestroyWindow(mhMainWnd);
	mhMainWnd = NULL;

	UnregisterClass(mAppName, mhAppInstance);
}

// Create status window
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

	// Create status window
	mStatusWnd = CreateWindow(L"EngineDemoStatusWnd", NULL, WS_VISIBLE | WS_OVERLAPPED, CW_USEDEFAULT, 0, 600, 300, NULL, NULL, mhAppInstance, NULL);

	if (!mStatusWnd)
	{
		return FALSE;
	}

	RECT R;
	GetClientRect(mStatusWnd, &R);
	// Create edit control
	mEdit = CreateWindow(WC_EDIT, L"    Status window initiated.", WS_VSCROLL | ES_READONLY | WS_HSCROLL | WS_VISIBLE | WS_CHILD | ES_MULTILINE, R.left, R.top, R.right, R.bottom, mStatusWnd, NULL, mhAppInstance, NULL);

	ShowWindow(mStatusWnd, SW_SHOW);
	UpdateWindow(mStatusWnd);

	return true;
}

void SystemClass::OnMouseDown(WPARAM btnState, int x, int y)
{
	mLastMousePos.x = x;
	mLastMousePos.y = y;

	SetCapture(mhMainWnd);
}

void SystemClass::OnMouseUp(WPARAM btnState, int x, int y)
{
	ReleaseCapture();
}

void SystemClass::OnMouseMove(WPARAM btnState, int x, int y)
{
	if ((btnState & MK_LBUTTON) != 0)
	{
		float dx = XMConvertToRadians(0.25f*static_cast<float>(x - mLastMousePos.x));
		float dy = XMConvertToRadians(0.25f*static_cast<float>(y - mLastMousePos.y));

		mTheta += dx;
		mPhi += dy;

		mPhi = min(max(mPhi, 0.1f), XM_PI - 0.1f);
	}
	else if ((btnState & MK_RBUTTON) != 0)
	{
		float dx = 0.05f*static_cast<float>(x - mLastMousePos.x);
		float dy = 0.05f*static_cast<float>(y - mLastMousePos.y);

		mRadius += dx - dy;

		mRadius = min(max(mRadius, 3.0f), 150.0f);
	}

	mLastMousePos.x = x;
	mLastMousePos.y = y;
}
