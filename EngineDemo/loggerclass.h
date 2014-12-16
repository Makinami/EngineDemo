#pragma once

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <CommCtrl.h>
#include <windowsx.h>

#include <string>
#include <fstream>
#include <time.h>


using namespace std;

// Input flags
#define LOG_NONE    0x00000000L
#define LOG_FILE	0x00000001L
#define LOG_WINDOW	0x00000002L
#define LOG_ALL		0xFFFFFFFFL

class LoggerClass
{
	public:
		LoggerClass(const std::wstring &fileName, HWND hWnd);
		~LoggerClass();

		int SetFile(const wstring &fileName);
		const wstring GetFile();

		int SetWindow(HWND hWnd);
		const HWND GetWindow();

		int Write(wstring msg, DWORD output = LOG_ALL);
		int Error(wstring msg, DWORD output = LOG_ALL);
		int Success(wstring msg, DWORD output = LOG_ALL);
		int Notice(wstring msg, DWORD output = LOG_ALL);

		DWORD GetValidChannels();

	private:
		inline void WriteFileRaw(wstring text);
		
		wofstream mOutFile;
		HWND	  mOutWnd;

		wstring mFileName;
		bool	mWndValid;
};