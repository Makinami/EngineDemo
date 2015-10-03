Texture2D<float3> transmittance : register(t0);
RWTexture2D<float3> deltaE : register(u0);

SamplerState samTransmittance : register(s0);

#define USE_TRANSMITTANCE

#pragma warning(disable:3568)
#include <resolutions.h>
#pragma warning(default:3568)

#include <common.hlsli>

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	float r, muS;
	getIrradianceRMuS(DTid.xy, r, muS);
	deltaE[DTid.xy] = getTransmittance(r, muS) * max(muS, 0.0);
}