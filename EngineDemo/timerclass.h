#pragma once

#define CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

class TimerClass
{
	public:
		TimerClass();

		float TotalTime() const; // sec
		float DeltaTime() const; // sec

		void Reset();
		void Start();
		void Stop();
		void Tick();

	private:
		double mSecondsPerCount;
		double mDeltaTime;

		__int64 mTotalTime;

		__int64 mStartTime;

		__int64 mPrevTime;
		__int64 mCurrTime;

		bool mStopped;
};