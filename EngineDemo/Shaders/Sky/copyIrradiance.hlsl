Texture2D<float3> deltaE : register(t0);
Texture2D<float3> irradiance : register(t1);

RWTexture2D<float3> copyIrradiance : register(u0);

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	copyIrradiance[DTid.xy] = deltaE[DTid.xy] + irradiance[DTid.xy];
}