#pragma once

#pragma warning(disable:4838)
#include <DirectXMath.h>
#pragma warning(default:4838)

#include <complex>

using namespace std;

class cFFT
{
public:
	cFFT(unsigned int N);
	~cFFT();

	unsigned int reverse(unsigned int i);
	complex<float> t(unsigned int x, unsigned int N);
	void fft(complex<float> *input, complex<float> *output, int stride, int offset);

private:
	unsigned int N, which;
	unsigned int log_2_N;
	unsigned int *reversed;
	complex<float> **T;
	complex<float> *c[2];
};