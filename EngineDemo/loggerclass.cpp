#include "loggerclass.h"

#include <codecvt>
#include <imgui.h>

// ------------------------------------------------------------------------
//                           LoggerClass definition
// ------------------------------------------------------------------------

inline Logger & Logger::Instance()
{
	static Logger _instance;
	return _instance;
}

Logger::Logger()
{
	SetFile(L"log_default.txt");
}

Logger::~Logger()
{
	outFile.close();
}

// Set new file output (while closing the previous one if existed)
int Logger::SetFile(const wstring &newName)
{
	if (fileName == newName)
	{
		Notice(L"Reset logger to the same file");
		return true;
	}

	wofstream newStream(newName);
	if (newStream.is_open())
	{
		if (outFile.is_open()) Notice(L"Logger moved to " + newName);
		outFile.close();
		outFile = move(newStream);
		fileName = newName;

		time_t timestamp = time(nullptr);
		tm timeinfo;
		localtime_s(&timeinfo, &timestamp);
		wchar_t date[50];
		wcsftime(date, 50, L"%x %X", &timeinfo);
		WriteRaw(outFile, L"    Logging initiated at: " + std::wstring(date) + L"\n\n");
		WriteRaw(outString, L"    Logging initiated at: " + std::wstring(date) + L"\n\n");

		return true;
	}
	else
	{
		Error(L"Couldn't set new log file: " + newName);
		return false;
	}
}

wstring Logger::GetFileName() const
{
	return fileName;
}

// Write message to log based on output (default: LOG_ALL - all available)
int Logger::Write(wstring msg, DWORD output)
{
	// Add current time to the beggining of the message
	time_t timestamp = time(nullptr);
	tm timeinfo;
	localtime_s(&timeinfo, &timestamp);
	wchar_t date[50];
	wcsftime(date, 50, L"%X", &timeinfo);
	std::wstring text = std::wstring(date) + L" : " + msg + L"\n";

	if ((output & LOG_FILE) && (fileName != L""))
	{
		// Write to file
		WriteRaw(outFile, text);
	}
	if (output & LOG_IMGUI)
	{
		WriteRaw(outString, text);
	}
	return 1;
}

// Append "ERROR: " and write
int Logger::Error(wstring msg, DWORD output)
{
	return Write(L"ERROR: " + msg);
}

// Append "success: " and write
int Logger::Success(wstring msg, DWORD output)
{
	return Write(L"Success: " + msg);
}

// Append "Notice: " and write
int Logger::Notice(wstring msg, DWORD output)
{
	return Write(L"Notice: " + msg);
}

void Logger::Render()
{
	static bool show = true;
	static const bool readonly = true;
	ImGui::SetNextWindowPos(ImVec2(0, 720-150));
	ImGui::SetNextWindowSize(ImVec2(1280, 150));
	ImGuiWindowFlags window_flags = 0;
	window_flags |= ImGuiWindowFlags_NoTitleBar;
	window_flags |= ImGuiWindowFlags_NoResize;
	window_flags |= ImGuiWindowFlags_NoMove;
	window_flags |= ImGuiWindowFlags_NoCollapse;
	ImGui::Begin("Log", &show, window_flags);

	ImGui::InputTextMultiline("##log", outString.str().data(), outString.str().size(), ImVec2(1264, 134), ImGuiInputTextFlags_ReadOnly);

	ImGui::End();
}

inline void Logger::WriteRaw(ostream& stream, wstring text)
{
	wstring_convert<codecvt_utf8<wchar_t>> myconv;
	stream << myconv.to_bytes(text) << flush;
}

// Write raw message to open file
inline void Logger::WriteRaw(wostream& stream, wstring text)
{
	stream << text << std::flush;
}


// ------------------------------------------------------------------------
//                           HasLogger definition
// ------------------------------------------------------------------------

HasLogger::HasLogger()
{}

HasLogger::~HasLogger()
{}

// Write to log
int HasLogger::LogWrite(wstring msg, DWORD output)
{
	return Logger::Instance().Write(msg, output);
}

// Write error to log
int HasLogger::LogError(wstring msg, DWORD output)
{
	return Logger::Instance().Error(msg, output);
}

// Write succes to log
int HasLogger::LogSuccess(wstring msg, DWORD output)
{
	return Logger::Instance().Success(msg, output);
}

// Write notice to log
int HasLogger::LogNotice(wstring msg, DWORD output)
{
	return Logger::Instance().Notice(msg, output);
}