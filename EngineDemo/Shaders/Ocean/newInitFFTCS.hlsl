#include "perFrameCB.hlsli" // b1

cbuffer constBuffer : register(b0)
{
	float4 INVERSE_GRID_SIZE;
	float4 GRID_SIZE;
	float4 MIN_K;
};

RWTexture2DArray<float> spectrumTex : register(u0);
Texture2DArray<float> phaseTex : register(t0);

RWTexture2DArray<float4> fftWaves : register(u1);

static const float PI = 3.14159265359;
static const float G = 9.81;
static const float KM = 370.0;

static const int FFT_SIZE = 256.0;

float2 multiplyComplex(float2 a, float2 b) {
	return float2(a[0] * b[0] - a[1] * b[1], a[1] * b[0] + a[0] * b[1]);
}

float2 multiplyByI(float2 z) {
	return float2(-z[1], z[0]);
}
float omega(float k) {
	return sqrt(G * k * (1.0 + k * k / KM * KM));
}

float getPhase(float2 coordinates, float n, float m)
{

}

float2 getSpectrum(int2 coordinates, float n, float m, int z)
{
	float2 waveVector = float2(2.0 * PI * float2(n, m)) * INVERSE_GRID_SIZE[z];

	float phase = phaseTex[uint3(coordinates,z)].r;
	phase += omega(length(waveVector)) * time;
	phase = fmod(phase, 2.0 * PI);

	float2 phaseVector;
	sincos(phase, phaseVector.x, phaseVector.y);

	float2 h0 = 0.0.xx, h0Star = 0.0.xx;
	h0.x = spectrumTex[uint3(coordinates, z)];
	h0Star.x = spectrumTex[uint3(255.0.xx - coordinates,z)];
	h0Star.y *= -1.0;

	float2 h = multiplyComplex(h0, phaseVector) + multiplyComplex(h0Star, float2(phaseVector.x, -phaseVector.y));

	return h;
}

[numthreads(16, 16, 1)]
void main( int3 DTid : SV_DispatchThreadID )
{
	float x = DTid.x >= FFT_SIZE * 0.5 ? DTid.x - FFT_SIZE : DTid.x;
	float y = DTid.y >= FFT_SIZE * 0.5 ? DTid.y - FFT_SIZE : DTid.y;

	float2 k[4];

	[unroll(4)]
	for (int i = 0; i < 4; ++i)
		k[i] = float2(x, y) * INVERSE_GRID_SIZE[i];
	/*float2 k1 = float2(x, y) * INVERSE_GRID_SIZE.x;
	float2 k2 = float2(x, y) * INVERSE_GRID_SIZE.y;
	float2 k3 = float2(x, y) * INVERSE_GRID_SIZE.z;
	float2 k4 = float2(x, y) * INVERSE_GRID_SIZE.w;*/

	float2 h[4];
	[unroll(4)]
	for (int i = 0; i < 4; ++i)
		h[i] = getSpectrum(DTid.xy, x, y, i);
	/*float2 h1 = getSpectrum(DTid.xy, 0);
	float2 h2 = getSpectrum(DTid.xy, 1);
	float2 h3 = getSpectrum(DTid.xy, 2);
	float2 h4 = getSpectrum(DTid.xy, 3);*/

	float2 slope[4];
	[unroll(4)]
	for (int i = 0; i < 4; ++i)
		slope[i] = multiplyByI(h[i]) * k[i];
	/*float2 slope1 = multiplyByI(h1) * k1;
	float2 slope2 = mulitplyByI(h2) * k1;*/

	float K;
	float IK[4];
	[unroll(4)]
	for (int i = 0; i < 4; ++i)
	{
		K = length(k[i]);
		IK[i] = K == 0.0 ? 0.0 : 1.0 / K;
	}

	[unroll(4)]
	for (int i = 0; i < 4; ++i)
	{
		fftWaves[uint3(DTid.xy, i)] = float4(slope[i] * IK[i], h[i]);
	}
	fftWaves[uint3(DTid.xy, 4)] = float4(slope[0], slope[1]);
	fftWaves[uint3(DTid.xy, 5)] = float4(slope[2], slope[3]);
}