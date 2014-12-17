#pragma once

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

class LoggerClass
{
	public:
		LoggerClass(const std::wstring &fileName, HWND hWnd);
		~LoggerClass();

		int SetFile(const wstring &fileName);
		wstring GetFile() const;

		int SetWindow(HWND hWnd);
		HWND GetWindow() const;

		int Write(wstring msg, DWORD output = LOG_ALL);
		int Error(wstring msg, DWORD output = LOG_ALL);
		int Success(wstring msg, DWORD output = LOG_ALL);
		int Notice(wstring msg, DWORD output = LOG_ALL);

		DWORD GetValidChannels() const;

	private:
		void WriteFileRaw(wstring text);
		
		wofstream mOutFile;
		HWND	  mOutWnd;

		wstring mFileName;
		bool	mWndValid;
};

class HasLogger
{
	public:
		HasLogger();
		~HasLogger();

		void SetLogger(std::shared_ptr<LoggerClass> &lLogger);

	protected:
		bool IsSet() const;
		DWORD GetValidChannels() const;

		int LogWrite(wstring msg, DWORD output = LOG_ALL);
		int LogError(wstring msg, DWORD output = LOG_ALL);
		int LogSuccess(wstring msg, DWORD output = LOG_ALL);
		int LogNotice(wstring msg, DWORD output = LOG_ALL);

		std::shared_ptr<LoggerClass> Logger;
};