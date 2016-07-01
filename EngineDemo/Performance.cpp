#include "Performance.h"
#include <Windows.h>
#include <stack>

// undefine windows min/max macros to use std functions
#undef min
#undef max

#include <algorithm>

namespace Debug
{
	// ------------------------------------------------------------------------
	//                       PerformanceClass definition
	// ------------------------------------------------------------------------

	PerformanceClass::PerformanceClass() :
		mFontWrapper(nullptr),
		mTextGeometry(0),
		mDWriteFactory(0)
	{
	}

	PerformanceClass::~PerformanceClass()
	{
		ReleaseCOM(mFontWrapper);
		ReleaseCOM(mTextGeometry);
		ReleaseCOM(mDWriteFactory);
	}

	// Initalize performance
	bool PerformanceClass::Init(ID3D11Device1 * device, ID3D11DeviceContext1 * mImmediateContext)
	{
		// Clock frequency
		__int64 tf;
		QueryPerformanceFrequency((LARGE_INTEGER*)&tf);
		clock = tf / 1000.0f;

		IFW1Factory* factory;
		// font factory
		FW1CreateFactory(FW1_VERSION, &factory);
		// font wrapper
		factory->CreateFontWrapper(device, L"Arial", &mFontWrapper);
		// text geometry
		factory->CreateTextGeometry(&mTextGeometry);
		// DirectWrite layout
//		mFontWrapper->GetDWriteFactory(&mDWriteFactory);
		
		return true;
	}

	// Draw statistics
	void PerformanceClass::Draw(ID3D11DeviceContext1 * mImmediateContext)
	{
		wchar_t buffer[256] = { 0 };
		for (unsigned int i = 0; i < Watch.size(); ++i)
		{
			_snwprintf_s(buffer, 256, L"%20s: %.3fms %.3fms %.3fms %.3fms", Watch[i].name.c_str(), Watch[i].curr/clock, Watch[i].min/clock, Watch[i].max/clock, Watch[i].avg/clock);
			//mFontWrapper->DrawString(mImmediateContext, buffer, 16.0f, 100.0f, 50.0f+i*20.0f, 0xff0099ff, FW1_NOGEOMETRYSHADER | FW1_RESTORESTATE);
		}
	}

	// Reserve new call name
	char PerformanceClass::ReserveName(std::wstring _name)
	{
		SType temp;
		temp.name = _name;
		temp.id = (char)Watch.size();
		temp.max = _I64_MIN;
		temp.min = _I64_MAX;
		temp.avg = 0;

		Watch.push_back(temp);

		return temp.id;
	}
	
	// Compute statistics every frame
	void PerformanceClass::Compute()
	{
		std::for_each(std::begin(Watch), std::end(Watch), [](SType& it) { it.curr = 0; });

		std::stack<SCall> tempStack;
		for (auto it : CallStack)
		{
			if (it.type == CallType::START)
				tempStack.push(it);
			else
			{
				while (tempStack.size() && it.id != tempStack.top().id)
				{
					Watch[tempStack.top().id].curr = -1;
					tempStack.pop();
				}

				if (tempStack.size())
				{
					Watch[tempStack.top().id].curr += it.time - tempStack.top().time;
					tempStack.pop();
				}
			}
		}

		CallStack.clear();

		for (auto&& it : Watch)
		{
			it.min = std::min(it.min, it.curr);
			it.max = std::max(it.max, it.curr);
			it.avg = it.avg * 199 / 200.0f + it.curr / 200.0f;
		}
	}

	// Reset statistics
	void PerformanceClass::ResetStats()
	{
		for (auto&& it : Watch)
		{
			it.avg = 0;
			it.max = _I64_MIN;
			it.min = _I64_MAX;
		}
	}

	// ------------------------------------------------------------------------
	//                           HasPerformance definition
	// ------------------------------------------------------------------------

	HasPerformance::HasPerformance() :
		Performance(nullptr)
	{
	}

	HasPerformance::~HasPerformance()
	{
	}

	void HasPerformance::SetPerformance(std::shared_ptr<PerformanceClass>& lPerformance)
	{
		Performance = lPerformance;
	}

	bool HasPerformance::IsSet() const
	{
		return Performance ? true : false;
	}
}