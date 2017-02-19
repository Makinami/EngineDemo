RWTexture3D<float4> noiseTex : register(u0);

cbuffer cbGenerateNoise : register(b0)
{
	uint4 cFrequency;
	uint textSize;
	uint3 pad;
}

#include "..\\PerlinNoise.hlsli"
#include "..\\WorleyNoise.hlsli"

// Utility function that maps a value from one range to another
float Remap(float original_value, float original_min, float original_max, float new_min, float new_max)
{
	return new_min + (((clamp(original_value, original_min, original_max) - original_min) /
		(original_max - original_min)) * (new_max - new_min));
}

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	float4 noise;
	float4 frequency = pow(2.0, cFrequency);
	[unroll(3)]
	for (uint i = 0; i < 3; ++i)
		noise[i] = 0.5*PerlinNoise(float3(DTid) / frequency[i] + 131.31*i, textSize.xxx / cFrequency[i]) + 0.5;

	noise.w = saturate(worley(float3(DTid), frequency[3], textSize / cFrequency[3]));

	noiseTex[DTid] = noise;
}