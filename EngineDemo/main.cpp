#define _CRTDBG_MAPALLOC

#include "systemclass.h"

int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ PSTR cmdLine, _In_ int showCmd)
{
	// Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
	
	SystemClass theSystem(hInstance);

	if (!theSystem.Init()) return 0;

	theSystem.Run();

	//ReportLiveObjects();
	return 0;
} 