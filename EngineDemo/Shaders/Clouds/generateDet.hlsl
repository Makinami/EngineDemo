RWTexture3D<float> clouds : register(u1);

#define TEXT_SIZE 32

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
	float f = fBmPerlin(DTid, 16.0)*0.5 + 0.5;
	f *= 1.0 - fBmWorley(DTid, 16.0, 0.5, 3);
	//f = Remap(f, 1.0 - fBmWorley(DTid, 16.0, 0.5, 3), 1.0, 0.0, 1.0);
	clouds[DTid] = dot(float3(1.0 - fBmWorley(DTid, 8.0, 0.5, 2), 1.0 - fBmWorley(DTid, 4.0, 0.5, 2), 1.0 - fBmWorley(DTid, 2.0, 0.5, 2))*0.75 + 0.25, float3(0.625, 0.25, 0.125));
}