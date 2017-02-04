#pragma once

#include <imgui.h>

#include <vector>

namespace ImGui
{
	bool Combo(const char* label, int* currIndex, std::vector<std::string>& values);

	bool ListBox(const char* label, int* currIndex, std::vector<std::string>& values);

	bool Button(const char* label, bool enabled, const float colour = 0.0, const ImVec2& size = ImVec2(0, 0));
}