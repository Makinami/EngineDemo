Texture2D<float4> deltaE : register(t0);
Texture2D<float4> irradiance : register(t1);

RWTexture2D<float4> copyIrradiance : register(u0);

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	copyIrradiance[DTid.xy] = deltaE[DTid.xy] + irradiance[DTid.xy];
}