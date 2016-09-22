Texture2D<float3> transmittance : register(t0);

RWTexture2D<float3> deltaE : register(u0);

SamplerState samBilinearClamp : register(s0);

#define USE_TRANSMITTANCE

#include "Common.hlsli"

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	float alt, szAngle;
	getIrradianceAltSzAngle(DTid.xy, alt, szAngle);
	float3 attenuation = getTransmittance(alt, szAngle);
	deltaE[DTid.xy] = attenuation * saturate(szAngle);
}