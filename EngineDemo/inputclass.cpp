#include "inputclass.h"

InputClass::InputClass()
: mDirectInput(0),
  mKeyboard(0),
  mMouse(0),
  mIsDraged(0),
  mlZ(0),
  mPrevlZ(0),
  mlZAcc(0)
{
	ZeroMemory(&mMouseState, sizeof(mMouseState));
	ZeroMemory(&mPrevMouseState, sizeof(mPrevMouseState));
	mDragStart = {-1,-1};
}

InputClass::~InputClass()
{
	Shutdown();
}

bool InputClass::Init(HINSTANCE hInstance, HWND hWnd, int screenWidth, int screenHeight)
{
	mScreenHeight = screenHeight;
	mScreenWidth = screenWidth;

	mMousePos = POINT{0,0};

	// Creating device
	if (FAILED(DirectInput8Create(hInstance, DIRECTINPUT_VERSION, IID_IDirectInput8, (void**)&mDirectInput, NULL))) return false;

	// Setting up keyboard
	if (FAILED(mDirectInput->CreateDevice(GUID_SysKeyboard, &mKeyboard, NULL))) return false;

	if (FAILED(mKeyboard->SetDataFormat(&c_dfDIKeyboard))) return false;

	if (FAILED(mKeyboard->SetCooperativeLevel(hWnd, DISCL_FOREGROUND|DISCL_NONEXCLUSIVE))) return false;

	if (FAILED(mKeyboard->Acquire())) return false;

	// Setting up mouse
	if (FAILED(mDirectInput->CreateDevice(GUID_SysMouse, &mMouse, NULL))) return false;

	if (FAILED(mMouse->SetDataFormat(&c_dfDIMouse))) return false;

	if (FAILED(mMouse->SetCooperativeLevel(hWnd, DISCL_FOREGROUND|DISCL_NONEXCLUSIVE))) return false;

	if (FAILED(mMouse->Acquire())) return false;

	return true;
}

void InputClass::Shutdown()
{
	if (mMouse)
	{
		mMouse->Unacquire();
		ReleaseCOM(mMouse);
	}

	if (mKeyboard)
	{
		mKeyboard->Unacquire();
		ReleaseCOM(mKeyboard);
	}

	ReleaseCOM(mDirectInput);
}

bool InputClass::Capture()
{
	mPrevMousePos = mMousePos;
	mPrevMouseState = mMouseState;
	mPrevlZ = mlZ;
	mlZ = mlZAcc;
	mlZAcc = 0;

	if (!ReadKeyboard()) return false;

	if (!ReadMouse()) return false;

	ProcessInput();

	return true;
}

POINT InputClass::GetMouseLocation()
{
	return mMousePos;
}

bool InputClass::ReadKeyboard()
{
	HRESULT hr;

	if (FAILED(hr = mKeyboard->GetDeviceState(sizeof(mKeyboardState), (LPVOID)&mKeyboardState)))
	{
		if (hr == DIERR_INPUTLOST || hr == DIERR_NOTACQUIRED) mKeyboard->Acquire();
		else return false;
	}

	return true;
	
}

bool InputClass::ReadMouse()
{
	HRESULT hr;

	if (FAILED(hr = mMouse->GetDeviceState(sizeof(DIMOUSESTATE), (LPVOID)&mMouseState)))
	{
		if (hr == DIERR_INPUTLOST || hr == DIERR_NOTACQUIRED) mMouse->Acquire();
		else return false;
	}

	return true;
}

void InputClass::ProcessInput()
{
	mMousePos.x += mMouseState.lX;
	mMousePos.y += mMouseState.lY;

	if (mMousePos.x < 0) mMousePos.x = 0;
	if (mMousePos.y < 0) mMousePos.y = 0;

	if (mMousePos.x > mScreenWidth) mMousePos.y = mScreenWidth;
	if (mMousePos.y > mScreenHeight) mMousePos.y = mScreenHeight;

	mClick = false;

	if (!mPrevMouseState.rgbButtons[0] && mMouseState.rgbButtons[0])
	{
		mDragStart = mMousePos;
	}
	else if (mPrevMouseState.rgbButtons[0] && mMouseState.rgbButtons[0])
	{
		mIsDraged = 1;
	}
	else if (mPrevMouseState.rgbButtons[0] && !mMouseState.rgbButtons[0])
	{
		if (mDragStart.x == mMousePos.x && mDragStart.y == mMousePos.y)
		{
			mClick = true;
		}
		else
		{
			mIsDraged = -1;
		}
	}
	else
	{
		mIsDraged = 0;
		mDragStart = {-1,-1};
	}
}

bool InputClass::IsKeyPressed(SHORT key)
{
	if (mKeyboardState[key] & 0x80) return true;
	else return false;
}

POINT InputClass::GetClick()
{
	if (mClick) return mMousePos;
	else return POINT{-1, -1};
}

bool InputClass::IsClick()
{
	return mClick;
}

SHORT InputClass::IsDrag()
{
	return mIsDraged;
}

POINT InputClass::GetDragDelta()
{
	return POINT{mMousePos.x - mPrevMousePos.x, mMousePos.y - mPrevMousePos.y};
}

RECT InputClass::GetDragWhole()
{
	return RECT{mDragStart.x, mDragStart.y, mMousePos.x, mMousePos.y};
}

void InputClass::PassZDelta(SHORT zDelta)
{
	mlZAcc += zDelta;
}

LONG InputClass::GetWheel()
{
	return mlZ;
}
