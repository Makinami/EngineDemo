#include "perFrameCB.hlsli" // b1

cbuffer constBuffer : register(b0)
{
	float4 INVERSE_GRID_SIZE;
	float4 GRID_SIZE;
};

Texture2DArray<float2> spectrum : register(t0);
RWTexture2D<float2> slopeVariances : register(u0);

#define FFT_SIZE 256
#define XM_PI 3.14159265

static const float SCALE = 10.0;
static const float N_SLOPE_VARIANCE = 16;

float2 getSlopeVariances(float2 k, float A, float B, float2 spectrum) {
	float w = 1.0 - exp(A * k.x * k.x + B * k.y * k.y);
	float2 kw = k * w;
	return kw * kw * dot(spectrum, spectrum) * 2.0;
}


[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	float A = pow(float(DTid.x) / (N_SLOPE_VARIANCE - 1.0), 4.0) * SCALE;
	float B = pow(float(DTid.y) / (N_SLOPE_VARIANCE - 1.0), 4.0) * SCALE;
	A = -0.5 * A;
	B = -0.5 * B;

	float2 slope = float2(0, 0);
	for (int y = 0; y < FFT_SIZE; ++y)
	{
		for (int x = 0; x < FFT_SIZE; ++x)
		{
			int i = x >= FFT_SIZE / 2 ? x - FFT_SIZE : x;
			int j = y >= FFT_SIZE / 2 ? y - FFT_SIZE : y;
			float2 k = 2.0 * XM_PI * float2(i, j);

			[unroll(4)]
			for (int i = 0; i < 4; ++i)
				slope += getSlopeVariances(k / GRID_SIZE[i], A, B, spectrum[uint3(x, y, i)]);
		}
	}

	slopeVariances[DTid.xy] = slope;
}