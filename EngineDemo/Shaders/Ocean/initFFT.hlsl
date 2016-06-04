#include "perFrameCB.hlsli" // b1

cbuffer constBuffer : register(b0)
{
	float4 INVERSE_GRID_SIZE;
	float4 GRID_SIZE;
};

Texture2DArray<float2> spectrum : register(t0);
RWTexture2DArray<float4> fftWaves : register(u0);

#define FFT_SIZE 256

float2 getSpectrum(float k, float2 s0, float2 s0c)
{
	float w = sqrt(9.81 * k * (1.0 + k * k / (370.0 * 370.0)));
	float c = cos(w * time);
	float s = sin(w * time);
	return float2((s0.x + s0c.x) * c - (s0.y + s0c.y) * s, (s0.x - s0c.x) * s + (s0.y - s0c.y) * c);
}

float2 i(float2 z) // i * z (complex)
{
	return float2(-z.y, z.x);
}

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	float x = DTid.x >= FFT_SIZE / 2 ? DTid.x - FFT_SIZE : DTid.x;
	float y = DTid.y >= FFT_SIZE / 2 ? DTid.y - FFT_SIZE : DTid.y;

	float4 spec = float4(spectrum[DTid], spectrum[uint3(FFT_SIZE - DTid.xy, DTid.z)]);

	float2 k = float2(x, y) * INVERSE_GRID_SIZE[DTid.z];

	float K = length(k);

	float IK = K == 0.0 ? 0.0 : 1.0 / K;

	float2 h = getSpectrum(K, spec.xy, spec.zw);

	float2 slope = i(k.x * h) - k.y * h;
	fftWaves[DTid] = float4(slope.xy * IK, h); // grid size i displacement xzy
}