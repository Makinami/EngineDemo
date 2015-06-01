#include "loggerclass.h"

// ------------------------------------------------------------------------
//                           LoggerClass definition
// ------------------------------------------------------------------------

// Initialize logger
LoggerClass::LoggerClass(const std::wstring &fileName, HWND hWnd)
: mOutWnd(hWnd),
  mFileName(fileName)
{
	SetFile(fileName);
	SetWindow(hWnd);
}

LoggerClass::~LoggerClass()
{
	mOutFile.close();
}

// Set new file output (while closing the previous one if existed)
int LoggerClass::SetFile(const wstring &fileName)
{
	mOutFile.close();
	mFileName = L"";
	mOutFile.open(fileName);
	if (mOutFile.is_open())
	{
		mFileName = fileName;
		time_t timestamp = time(nullptr);
		tm timeinfo;
		localtime_s(&timeinfo, &timestamp);
		wchar_t date[50];
		wcsftime(date, 50, L"%x %X", &timeinfo);
		WriteFileRaw(L"    Logging initiated at: " + std::wstring(date) + L"\n\n");
		return true;
	}
	else
	{
		mOutFile.close();
		mFileName = L"";
		return false;
	}
}

wstring LoggerClass::GetFile() const
{
	return mFileName;
}

// Set new output Edit Control
int LoggerClass::SetWindow(HWND hWnd)
{
	wchar_t lpClassName[20];
	GetClassName(hWnd, lpClassName, 20);
	if (!std::wcscmp(lpClassName, WC_EDIT))
	{
		mOutWnd = hWnd;
		mWndValid = true;
		return true;
	}
	else
	{
		mOutWnd = 0;
		mWndValid = false;
		return false;
	}
}

HWND LoggerClass::GetWindow() const
{
	return mOutWnd;
}

// Write message to log based on output (default: LOG_ALL - all available)
int LoggerClass::Write(wstring msg, DWORD output)
{
	if ((output & LOG_WINDOW) && (mOutWnd))
	{
		std::wstring text = L"\r\n" + msg;
		auto selection = Edit_GetSel(mOutWnd); // save current selection
		auto length = Edit_GetTextLength(mOutWnd);
		SendMessage(mOutWnd, EM_SETSEL, (WPARAM)length, (LPARAM)length); // set coursor at the end of edit
		SendMessage(mOutWnd, EM_REPLACESEL, 0, (LPARAM)text.c_str()); // write message
		SendMessage(mOutWnd, EM_SETSEL, (WPARAM)LOWORD(selection), (LPARAM)HIWORD(selection)); // restore selection
	}
	if ((output & LOG_FILE) && (mFileName != L""))
	{
		// Add current time to the beggining of the message
		time_t timestamp = time(nullptr);
		tm timeinfo;
		localtime_s(&timeinfo, &timestamp);
		wchar_t date[50];
		wcsftime(date, 50, L"%X", &timeinfo);
		std::wstring text = std::wstring(date) + L" : " + msg + L"\n";
		// Write to file
		WriteFileRaw(text);
	}
	return 1;
}

// Append "ERROR: " and write
int LoggerClass::Error(wstring msg, DWORD output)
{
	return Write(L"ERROR: " + msg);
}

// Append "success: " and write
int LoggerClass::Success(wstring msg, DWORD output)
{
	return Write(L"success: " + msg);
}

// Append "Notice: " and write
int LoggerClass::Notice(wstring msg, DWORD output)
{
	return Write(L"Notice: " + msg);
}

// Get valid output channels
DWORD LoggerClass::GetValidChannels() const
{
	DWORD validChannels = LOG_NONE;
	if (mOutWnd) validChannels |= LOG_WINDOW;
	if (mFileName != L"") validChannels |= LOG_FILE;
	
	return validChannels;
}

// Write raw message to open file
inline void LoggerClass::WriteFileRaw(wstring text)
{
	if (mFileName != L"") mOutFile << text << std::flush;
}


// ------------------------------------------------------------------------
//                           HasLogger definition
// ------------------------------------------------------------------------

HasLogger::HasLogger()
{}

HasLogger::~HasLogger()
{}

// Check if logger set
bool HasLogger::IsSet() const
{
	return Logger ? true : false;
}


// Retrieve valid output channels
DWORD HasLogger::GetValidChannels() const
{
	if (IsSet()) return Logger->GetValidChannels();
	else return LOG_NONE;
}

// Write to log
int HasLogger::LogWrite(wstring msg, DWORD output)
{
	return IsSet() ? Logger->Write(msg, output) : 0;
}

// Write error to log
int HasLogger::LogError(wstring msg, DWORD output)
{
	return IsSet() ? Logger->Error(msg, output) : 0;
}

// Write succes to log
int HasLogger::LogSuccess(wstring msg, DWORD output)
{
	return IsSet() ? Logger->Success(msg, output) : 0;
}

// Write notice to log
int HasLogger::LogNotice(wstring msg, DWORD output)
{
	return IsSet() ? Logger->Notice(msg, output) : 0;
}

// Set new log
void HasLogger::SetLogger(std::shared_ptr<LoggerClass>& lLogger)
{
	Logger = lLogger;
}