RWTexture3D<float> clouds : register(u0);

#include "..\\PerlinNoise.hlsli"
#include "..\\WorleyNoise.hlsli"

// Utility function that maps a value from one range to another
float Remap(float original_value, float original_min, float original_max, float new_min, float new_max)
{
	return new_min + (((clamp(original_value, original_min, original_max) - original_min) /
		(original_max - original_min)) * (new_max - new_min));
}

float testFBM(float3 p)
{
	float n = 0.0;
	float freq = 16.0;
	float amp = 1.0;

	for (int i = 0; i < 5; ++i)
	{
		n += PerlinNoise(p / freq + 131.31*i, uint3(128, 128, 128) / freq)*amp;
		freq /= 2.5789;
		amp *= 0.707;
	}

	return n;
}

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	float p = fBmPerlin(DTid, 16.0)*0.5 + 0.5;
	float w = 1.0 - fBmWorley(DTid, 16.0, 0.5, 3);
	clouds[DTid] = p;
	return;
	float dis = length(64.0f.xxx - DTid);
	
	//clouds[DTid] = 2.0*(1.0 - smoothstep(20.0, 64.0, dis)) - 1.0 + (Remap(p, -(1.0 - w), 1.0, 0.0, 1.0) - 0.45)*5.0 / 2.0;
	clouds[DTid] = Remap(p, -w, 1.0, 0.0, 1.0);
}