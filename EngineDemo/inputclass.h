#pragma once

#pragma comment(lib, "dinput8.lib")
#pragma comment(lib, "dxguid.lib")

/* temp */
//---------------------------------------------------------------------------------------
// Convenience macro for releasing COM objects.
//---------------------------------------------------------------------------------------

#define ReleaseCOM(x) { if(x){ x->Release(); x = 0; } }

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#define DIRECTINPUT_VERSION 0x0800
#include <dinput.h>

class InputClass
{
	public:
		InputClass();
		~InputClass();

		bool Init(HINSTANCE hInstance, HWND hWnd, int screenWidth, int screenHeight);
		void Shutdown();
		bool Capture();

		POINT GetMouseLocation();

		bool IsKeyPressed(SHORT key);

	private:
		bool ReadKeyboard();
		bool ReadMouse();
		void ProcessInput();

		IDirectInput8* mDirectInput;
		IDirectInputDevice8* mKeyboard;
		IDirectInputDevice8* mMouse;

		unsigned char mKeyboardState[256];
		DIMOUSESTATE2 mMouseState;

		int mScreenWidth, mScreenHeight;
		POINT mMousePos;
};