#include "inputclass.h"

#include "imgui.h"

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
	mDrag = { 0,0 };
	mCapturedDrag = { 0,0 };
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

	mhWnd = hWnd;

	return true;
}

void InputClass::Shutdown()
{
}

bool InputClass::Capture()
{
	mCapturedDrag = mDrag;
	mDrag = { 0,0 };

	mWholeDrag.x += mCapturedDrag.x;
	mWholeDrag.y += mCapturedDrag.y;

	if (mCapturedDrag.x || mCapturedDrag.y) mIsDraged = true;
	else mIsDraged = false;

	mCapturedMousePos = mLastMousePos;

	return true;
}

void InputClass::OnMouseDown(WPARAM btnState, int x, int y, float time)
{
	mLastMousePos.x = x;
	mLastMousePos.y = y;

	mDragStart = mLastMousePos;
	mWholeDrag = { 0,0 };

	if (btnState & MK_LBUTTON)
	{
		mLeftDownTime = time;
		mLeftDown = true;
	}
	else if (btnState & MK_RBUTTON)
	{
		mRightDownTime = time;
		mRightDown = true;
	}

	SetCapture(mhWnd);
}

void InputClass::OnMouseUp(WPARAM btnState, int x, int y, float time)
{
	if (btnState & MK_LBUTTON)
	{
		mLeftDown = false;
		if (time - mLeftDownTime <= 0.25f)
		{
			mClicks.push(ClickEvent{ btnState, {x,y}, time });
		}
	}
	else if (btnState & MK_RBUTTON)
	{
		mRightDown = false;
		if (time - mRightDownTime <= 0.25f)
			mClicks.push(ClickEvent{ btnState, {x,y}, time });
	}

	ReleaseCapture();
}

void InputClass::OnMouseMove(WPARAM btnState, int x, int y)
{
	if (btnState & MK_LBUTTON)
	{
		mDrag.x += x - mLastMousePos.x;
		mDrag.y += y - mLastMousePos.y;
	}

	mLastMousePos = { x, y };
}

POINT InputClass::GetMouseLocation()
{
	return mCapturedMousePos;
}

bool InputClass::ReadKeyboard()
{

	return true;
	
}

bool InputClass::ReadMouse()
{
	return true;
}

void InputClass::ProcessInput()
{
}

bool InputClass::IsKeyPressed(SHORT key)
{
	if ((GetAsyncKeyState(key) & 0x8000) && !ImGui::GetIO().WantCaptureKeyboard) return true;
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
	return mCapturedDrag;
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
