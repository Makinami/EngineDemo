#pragma once

#define CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

#pragma comment(lib, "dinput8.lib")
#pragma comment(lib, "dxguid.lib")

/* temp */
//---------------------------------------------------------------------------------------
// Convenience macro for releasing COM objects.
//---------------------------------------------------------------------------------------

#define ReleaseCOM(x) { if(x){ x->Release(); x = nullptr; } }

#include <queue>

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#define DIRECTINPUT_VERSION 0x0800
#include <dinput.h>

class InputClass
{
	public:
		struct ClickEvent
		{
			WPARAM btnState;
			POINT Pos;
			float time;
		};

	public:
		InputClass();
		~InputClass();

		bool Init(HINSTANCE hInstance, HWND hWnd, int screenWidth, int screenHeight);
		void Shutdown();
		bool Capture();

		void OnMouseDown(WPARAM btnState, int x, int y, float time);
		void OnMouseUp(WPARAM btnState, int x, int y, float time);
		void OnMouseMove(WPARAM btnState, int x, int y);

		POINT GetMouseLocation();

		bool IsKeyPressed(SHORT key);

		POINT GetClick();
		bool IsClick();

		SHORT IsDrag();
		POINT GetDragDelta();
		RECT GetDragWhole();

		void PassZDelta(SHORT zDelta);
		LONG GetWheel();

	private:
		POINT mLastMousePos;
		POINT mCapturedMousePos;
		POINT mDrag;
		POINT mCapturedDrag;
		POINT mWholeDrag;

		float mLeftDownTime;
		float mRightDownTime;

		bool mLeftDown;
		bool mRightDown;

		HWND mhWnd;

		std::queue<ClickEvent> mClicks;


		// unnecesary
		bool ReadKeyboard();
		bool ReadMouse();
		void ProcessInput();

		IDirectInput8* mDirectInput;
		IDirectInputDevice8* mKeyboard;
		IDirectInputDevice8* mMouse;

		unsigned char mKeyboardState[256];
		DIMOUSESTATE mMouseState;
		LONG mlZ;

		int mScreenWidth, mScreenHeight;
		POINT mMousePos;
		LONG mlZAcc;

		DIMOUSESTATE mPrevMouseState;
		LONG mPrevlZ;
		POINT mPrevMousePos;
		POINT mDragStart;
		SHORT mIsDraged;
		bool mClick;
};