#pragma once

#define CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <CommCtrl.h>
#include <windowsx.h>

#include <string>
#include <fstream>
#include <time.h>
#include <memory>


using namespace std;

// Input flags
#define LOG_NONE    0x00000000L
#define LOG_FILE	0x00000001L
#define LOG_WINDOW	0x00000002L
#define LOG_ALL		0xFFFFFFFFL

/*
	Main logger class:
		Has the ability to both log to external file, and earlier defined Window's Edit Control.
*/
class LoggerClass
{
	public:
		LoggerClass(const std::wstring &fileName = L"log.txt", HWND hWnd = 0);
		~LoggerClass();

		// Set/get filename for external logger
		int SetFile(const wstring &fileName);
		wstring GetFile() const;

		// Set/get edit control handler
		int SetWindow(HWND hWnd);
		HWND GetWindow() const;

		//Write to log
		int Write(wstring msg, DWORD output = LOG_ALL);

		// Shortucts for appending respectively "ERROR: ", "succes:", "Notice: " to the beggining of log.
		int Error(wstring msg, DWORD output = LOG_ALL);
		int Success(wstring msg, DWORD output = LOG_ALL);
		int Notice(wstring msg, DWORD output = LOG_ALL);

		// Retrieve defined and valid output channels
		DWORD GetValidChannels() const;

	private:
		// Output raw message to file
		void WriteFileRaw(wstring text);
		
		// Output
		wofstream mOutFile;
		HWND	  mOutWnd;

		// Shortcuts for chaking validity (as well as filename)
		wstring mFileName;
		bool	mWndValid;
};


/*
	Logger possesion class:
		Works as a base class for other modules for easier adding and operating logger
*/
class HasLogger
{
	public:
		HasLogger();
		~HasLogger();

		// Set previously 'globaly' (or not) created Logger
		void SetLogger(std::shared_ptr<LoggerClass> &lLogger);

	protected:
		// Check is Logger set
		bool IsSet() const;
		// Check valid outputs
		DWORD GetValidChannels() const;

		// Shortcuts to writing to logger
		int LogWrite(wstring msg, DWORD output = LOG_ALL);
		int LogError(wstring msg, DWORD output = LOG_ALL);
		int LogSuccess(wstring msg, DWORD output = LOG_ALL);
		int LogNotice(wstring msg, DWORD output = LOG_ALL);

		// Shared logger
		std::shared_ptr<LoggerClass> Logger;
};