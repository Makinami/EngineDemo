Texture2D<float3> transmittance : register(t0);

Texture3D<float4> deltaSR : register(t2);
Texture3D<float4> deltaSM : register(t3);

RWTexture2D<float4> deltaE : register(u0);

SamplerState samTransmittance : register(s0);
SamplerState samDeltaSR : register(s1);
SamplerState samDeltaSM : register(s2);

#define USE_TRANSMITTANCE
#define USE_DELTAS

#pragma warning(disable:3568)
#include <resolutions.h>
#pragma warning(default:3568)

#include <common.hlsli>

cbuffer Order : register(b0)
{
	int4 order;
};

static const float dphi = PI / float(IRRADIANCE_INTEGRAL_SAMPLES);
static const float dtheta = PI / float(IRRADIANCE_INTEGRAL_SAMPLES);

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	float r, muS;
	getIrradianceRMuS(DTid.xy, r, muS);
	float3 s = float3(max(sqrt(1.0 - muS*muS), 0.0), 0.0, muS);

	float3 result = float3(0.0, 0.0, 0.0);
	// integral over 2.PI around x with two nested loops over w directions (theta, phi) -- Eq (15)
	for (int iphi = 0; iphi < 2 * IRRADIANCE_INTEGRAL_SAMPLES; ++iphi)
	{
		float phi = (float(iphi) + 0.5) * dphi;
		for (int itheta = 0; itheta < IRRADIANCE_INTEGRAL_SAMPLES / 2; ++itheta)
		{
			float theta = (float(itheta) + 0.5) * dtheta;
			float dw = dtheta * dphi * sin(theta);
			float3 w = float3(cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta));
			float nu = dot(s, w);
			if (order.x == 2)
			{
				// first iteration is special because Rayleigh and Mie ware stored separately,
				// without the phase factors; they must be reintroduced here
				float pr1 = phaseFunctionR(nu);
				float pm1 = phaseFunctionM(nu);
				float3 ray1 = getDeltaSR(r, w.z, muS, nu);
				float3 mie1 = getDeltaSM(r, w.z, muS, nu);
				result += (ray1*pr1 + mie1*pm1)*w.z*dw;
			}
			else
			{
				result += getDeltaSR(r, w.z, muS, nu) * w.z * dw;
			}
		}
	}

	deltaE[DTid.xy] = float4(result, 0.0);
}