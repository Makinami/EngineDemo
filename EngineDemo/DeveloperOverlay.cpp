#include "DeveloperOverlay.h"

namespace ImGui
{
	static auto vector_getter = [](void* vec, int idx, const char** out_text)
	{
		auto& vector = *static_cast<std::vector<std::string>*>(vec);
		if (idx < 0 || idx >= static_cast<int>(vector.size())) { return false; }
		*out_text = vector.at(idx).c_str();
		return true;
	};

	bool Combo(const char* label, int* currIndex, std::vector<std::string>& values)
	{
		if (values.empty()) { return false; }
		return Combo(label, currIndex, vector_getter,
			static_cast<void*>(&values), values.size());
	}

	bool ListBox(const char* label, int* currIndex, std::vector<std::string>& values)
	{
		if (values.empty()) { return false; }
		return ListBox(label, currIndex, vector_getter,
			static_cast<void*>(&values), values.size());
	}

	bool Button(const char* label, bool enabled, const float colour, const ImVec2& size)
	{
		if (enabled)
		{
			ImGui::PushStyleColor(ImGuiCol_Button, ImColor::HSV(colour, 0.6f, 0.6f));
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImColor::HSV(colour, 0.7f, 0.7f));
			ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImColor::HSV(colour, 0.8f, 0.8f));
		}
		else
		{
			ImGui::PushStyleColor(ImGuiCol_Button, ImColor::HSV(colour, 0.09f, 0.28f));
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImColor::HSV(colour, 0.09f, 0.28f));
			ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImColor::HSV(colour, 0.09f, 0.28f));
		}

		auto clicked = ImGui::Button(label);

		ImGui::PopStyleColor(3);

		return enabled && clicked;
	}
}