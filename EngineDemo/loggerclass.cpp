#include "loggerclass.h"

LoggerClass::LoggerClass(const std::wstring &fileName = L"log.txt", HWND hWnd = 0)
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

int LoggerClass::SetFile(const wstring &fileName)
{
	mOutFile.close();
	mFileName = L"";
	mOutFile.open(fileName);
	if (mOutFile.is_open())
	{
		mFileName = fileName;
		time_t timestamp = time(nullptr);
		auto timeinfo = localtime(&timestamp);
		wchar_t date[50];
		wcsftime(date, 50, L"%x %X", timeinfo);
		WriteFileRaw(L"    Logging initiated at: " + std::wstring(date) + L"\n\n");
		return 0;
	}
	else
	{
		mOutFile.close();
		mFileName = L"";
		return 1;
	}
}

wstring LoggerClass::GetFile() const
{
	return mFileName;
}

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

int LoggerClass::Write(wstring msg, DWORD output)
{
	if ((output & LOG_WINDOW) && (mOutWnd))
	{
		std::wstring text = L"\r\n" + msg;
		auto selection = Edit_GetSel(mOutWnd);
		auto length = Edit_GetTextLength(mOutWnd);
		SendMessage(mOutWnd, EM_SETSEL, (WPARAM)length, (LPARAM)length);
		SendMessage(mOutWnd, EM_REPLACESEL, 0, (LPARAM)text.c_str());
		SendMessage(mOutWnd, EM_SETSEL, (WPARAM)LOWORD(selection), (LPARAM)HIWORD(selection));
	}
	if ((output & LOG_FILE) && (mFileName != L""))
	{
		time_t timestamp = time(nullptr);
		auto timeinfo = localtime(&timestamp);
		wchar_t date[50];
		wcsftime(date, 50, L"%X", timeinfo);
		std::wstring text = std::wstring(date) + L" : " + msg + L"\n";
		WriteFileRaw(text);
	}
	return 1;
}

int LoggerClass::Error(wstring msg, DWORD output)
{
	return Write(L"ERROR: " + msg);
}

int LoggerClass::Success(wstring msg, DWORD output)
{
	return Write(L"success: " + msg);
}

int LoggerClass::Notice(wstring msg, DWORD output)
{
	return Write(L"Notice: " + msg);
}

DWORD LoggerClass::GetValidChannels() const
{
	DWORD validChannels = LOG_NONE;
	if (mOutWnd) validChannels |= LOG_WINDOW;
	if (mFileName != L"") validChannels |= LOG_FILE;
	
	return validChannels;
}

inline void LoggerClass::WriteFileRaw(wstring text)
{
	mOutFile << text << std::flush;
}


HasLogger::HasLogger()
{}

HasLogger::~HasLogger()
{}

bool HasLogger::IsSet() const
{
	return Logger ? true : false;
}

DWORD HasLogger::GetValidChannels() const
{
	if (IsSet()) return Logger->GetValidChannels();
	else return LOG_NONE;
}

int HasLogger::LogWrite(wstring msg, DWORD output)
{
	return IsSet() ? Logger->Write(msg, output) : 0;
}

int HasLogger::LogError(wstring msg, DWORD output)
{
	return IsSet() ? Logger->Error(msg, output) : 0;
}

int HasLogger::LogSuccess(wstring msg, DWORD output)
{
	return IsSet() ? Logger->Success(msg, output) : 0;
}

int HasLogger::LogNotice(wstring msg, DWORD output)
{
	return IsSet() ? Logger->Notice(msg, output) : 0;
}

void HasLogger::SetLogger(std::shared_ptr<LoggerClass>& lLogger)
{
	Logger = lLogger;
}