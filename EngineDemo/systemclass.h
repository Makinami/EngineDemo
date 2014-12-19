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

/*
Main System Class:
Creates new game and status window (at least for now) and manages other subsystems
*/
class SystemClass
{
	public:
		SystemClass(HINSTANCE hInstance);
		SystemClass(const SystemClass&);
		~SystemClass();

		// Initiate all systems based on setting from file
		bool Init(std::string filename = "settings.ini");
		// Shutdown main system and shut down and delete all subsystems
		void Shutdown();
		// Initiate main loop
		int Run();

		// Message proc for main window
		LRESULT CALLBACK MsgProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
		// Message proc for status window
		LRESULT CALLBACK StatusWndMsgProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

		private:
		// Run each frame
		bool Frame();

		// Init/shutdown main window
		bool InitMainWindow();
		void ShutdownMainWindow();

		// Create status window
		bool CreateStatusWindow();

	private:
		// Application and main window properties
		HINSTANCE	 mhAppInstance;
		HWND		 mhMainWnd;
		LPCWSTR		 mAppName;
		std::wstring mWndCap;
		bool		 mAppPaused;

		enum
		{
			WndStateNormal, WndStateMaximized, WndStateFullScreen, WndStateMinimized, WndStateResizing
		} mWndState;

		// Main window/game client size
		int mClientWidth;
		int mClientHeight;

		// Logger windows handlers
		HWND mStatusWnd;
		HWND mEdit;

		/*
		Subsystems
		*/
		std::shared_ptr<D3DClass> mD3D; // Main DirectX 3D

		std::shared_ptr<INIReader> Settings; // Setting
		std::shared_ptr<LoggerClass> Logger; // Logger
};