#pragma once

#if defined(DEBUG) || defined(_DEBUG)
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#endif

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include <memory>

#include "inih\cpp\INIReader.h"

#include "loggerclass.h"
#include "d3dclass.h"

class SystemClass
{
	public:
		SystemClass(HINSTANCE hInstance);
		SystemClass(const SystemClass&);
		~SystemClass();

		bool Init(std::string filename);
		void Shutdown();
		int Run();

		LRESULT CALLBACK MsgProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

	private:
		bool Frame();
		bool InitMainWindow();
		void ShutdownMainWindow();

	private:
		HINSTANCE	 mhAppInstance;
		HWND		 mhMainWnd;
		LPCWSTR		 mAppName;
		std::wstring mWndCap;

		std::shared_ptr<D3DClass> mD3D;

		int mClientWidth;
		int mClientHeight;

		std::shared_ptr<INIReader> Settings;
		std::shared_ptr<LoggerClass> Logger;
};