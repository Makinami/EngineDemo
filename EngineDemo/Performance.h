#pragma once

#include <d3d11_1.h>
#include <stack>
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <list>

#define ReleaseCOM(x) { if(x){ x->Release(); x = nullptr; } }
#define SafeDelete(x) { delete x; x = nullptr; }

#include "FW1FontWrapper.h"
#pragma comment (lib, "FW1FontWrapper.lib")

/*
	Debug namespace:
		Contains all code related to debuging, profiling, etc.

*/
namespace Debug
{
	/*
		PerformaceClass:
			Keep track of the profiling START/END calls and draw statistics.
	*/
	class PerformanceClass
	{
	public:
		PerformanceClass();
		~PerformanceClass();

		bool Init(ID3D11Device1* device, ID3D11DeviceContext1* mImmediateContext);

		// Register call with a timestamp
		enum class CallType { START, END };
		inline void Call(char _id, CallType _call)
		{
			__int64 callTime;
			QueryPerformanceCounter((LARGE_INTEGER*)&callTime);

			CallStack.push_back(SCall{ _id, callTime, _call });
		};

		// Draw statistics
		void Draw(ID3D11DeviceContext1* mImmediateContext);

		// Register new call name
		char ReserveName(std::wstring _name);

		// Count statistics
		void Compute();
		void ResetStats();

	private:
		// Call data
		struct SCall {
			char id;
			__int64 time;
			CallType type;
		};

		// Registered call name's data
		struct SType {
			char id;
			std::wstring name;
			__int64 curr;
			__int64 min;
			__int64 max;
			float avg;
		};

		// Per frame call queue
		std::list<SCall> CallStack;
		// All registered call name's data
		std::vector<SType> Watch;

	private:
		float clock;
		// FW1 Font Wrapper for displaying statistics
		IFW1FontWrapper* mFontWrapper;
		IFW1TextGeometry* mTextGeometry;
		IDWriteFactory* mDWriteFactory;
	};

	/*
		HasPerformance class:
			Basic interface for any module which will use performance profiling
	*/
	class HasPerformance
	{
	public:
		HasPerformance();
		~HasPerformance();

		// Set previously initiated Performance
		void SetPerformance(std::shared_ptr<PerformanceClass> &lPerformance);

		inline void CallStart(char id) { Performance->Call(id, Debug::PerformanceClass::CallType::START); };
		inline void CallEnd(char id) { Performance->Call(id, Debug::PerformanceClass::CallType::END); };

	protected:
		// Check is Performance set
		bool IsSet() const;

		// Shared Performance
		std::shared_ptr<Debug::PerformanceClass> Performance;
	};
}