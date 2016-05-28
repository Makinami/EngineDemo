Texture3D<float> seed;
RWTexture3D<float4> clouds;

#include "SimplexNoise.hlsli"

float WorleyNoise(float3 x, float3 period);

[numthreads(16, 1, 16)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	clouds[DTid] = abs(snoise(DTid/16.0f));
}