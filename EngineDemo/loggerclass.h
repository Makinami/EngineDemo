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
#include <sstream>
#include <time.h>
#include <memory>


using namespace std;

// Input flags
#define LOG_NONE    0x00000000L
#define LOG_FILE	0x00000001L
#define LOG_WINDOW	0x00000002L
#define LOG_IMGUI	0x00000004L
#define LOG_ALL		0xFFFFFFFFL

/*
	Main logger class:
		Has the ability to both log to external file, and earlier defined Window's Edit Control.
*/
class Logger
{
	public:
		static Logger& Instance();

		// Set/get filename for external logger
		int SetFile(const wstring &fileName);
		wstring GetFileName() const;

		//Write to log
		int Write(wstring msg, DWORD output = LOG_ALL);

		// Shortucts for appending respectively "ERROR: ", "succes:", "Notice: " to the beggining of log.
		int Error(wstring msg, DWORD output = LOG_ALL);
		int Success(wstring msg, DWORD output = LOG_ALL);
		int Notice(wstring msg, DWORD output = LOG_ALL);

		void Render();

	private:
		Logger();
		Logger(const Logger&) = delete;
		Logger& operator=(const Logger&) = delete;
		~Logger();

		// Output raw message to file
		void WriteRaw(ostream& stream, wstring text);
		void WriteRaw(wostream& stream, wstring text);
		
		// Output
		wofstream outFile;
		ostringstream outString;

		// Shortcuts for chaking validity (as well as filename)
		wstring fileName; // default: log_default.txt
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

	protected:
		// Shortcuts to writing to logger
		int LogWrite(wstring msg, DWORD output = LOG_ALL);
		int LogError(wstring msg, DWORD output = LOG_ALL);
		int LogSuccess(wstring msg, DWORD output = LOG_ALL);
		int LogNotice(wstring msg, DWORD output = LOG_ALL);
};