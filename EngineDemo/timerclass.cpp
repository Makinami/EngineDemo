#include <Windows.h>
#include "timerclass.h"

TimerClass::TimerClass()
: mSecondsPerCount(0.0), mDeltaTime(0.0), mPrevTime(0), mCurrTime(0), mStopped(false)
{
	__int64 countsPerSec;
	QueryPerformanceFrequency((LARGE_INTEGER*)&countsPerSec);
	mSecondsPerCount = 1.0 / (double)countsPerSec;
}

float TimerClass::TotalTime() const
{
	return (float)mTotalTime*mSecondsPerCount;
}

float TimerClass::DeltaTime() const
{
	return mDeltaTime;
}

void TimerClass::Reset()
{
	__int64 currTime;
	QueryPerformanceCounter((LARGE_INTEGER*)&currTime);

	mTotalTime = 0;
	mStartTime = currTime;
	mPrevTime = currTime;
	mCurrTime = currTime;
	mStopped = 0;
	mDeltaTime = 0.0f;
}

void TimerClass::Start()
{
	if (mStopped)
	{
		__int64 startTime;
		QueryPerformanceCounter((LARGE_INTEGER*)&startTime);

		mPrevTime += startTime - mCurrTime;
		mCurrTime = startTime;
		mStopped = false;
	}
}

void TimerClass::Stop()
{
	if ( !mStopped )
	{
		__int64 currTime;
		QueryPerformanceCounter((LARGE_INTEGER*)&currTime);

		mCurrTime = currTime;
		mStopped = true;
	}
}

void TimerClass::Tick()
{
	if ( mStopped )
	{
		mDeltaTime = 0.0;
		return;
	}

	QueryPerformanceCounter((LARGE_INTEGER*)&mCurrTime);

	mDeltaTime = (mCurrTime - mPrevTime)*mSecondsPerCount;
	mTotalTime += (mCurrTime > mPrevTime) ? (mCurrTime - mPrevTime) : 0;

	mPrevTime = mCurrTime;

	// non negative. 
	if ( mDeltaTime < 0.0f )
	{
		mDeltaTime = 0.0f;
	}
}
