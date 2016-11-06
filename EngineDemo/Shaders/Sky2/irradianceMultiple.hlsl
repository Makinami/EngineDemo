Texture3D<float4> deltaSR : register(t0);
Texture3D<float4> deltaSM : register(t1);

RWTexture2D<float4> deltaE : register(u0);

#define USE_DELTAS

#include "Common.hlsli"

cbuffer Order : register(b0)
{
	int4 order;
};

static const float dphi = PI / float(IRRADIANCE_INTEGRAL_SAMPLES);
static const float dtheta = PI / float(IRRADIANCE_INTEGRAL_SAMPLES);

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	float alt, szAngle;
	getIrradianceAltSzAngle(DTid.xy, alt, szAngle);
	float3 s = float3(max(sqrt(1.0 - szAngle*szAngle), 0.0), 0.0, szAngle);

	float3 result = 0.0.xxx;

	// integral over 2PI around current position with two nested loops over w direction (theta, phi)
	for (int iphi = 0; iphi < 2 * IRRADIANCE_INTEGRAL_SAMPLES; ++iphi)
	{
		float phi = (float(iphi) + 0.5)*dphi;
		for (int itheta = 0; itheta < IRRADIANCE_INTEGRAL_SAMPLES / 2; ++itheta)
		{
			float theta = (float(itheta) + 0.5) * dtheta;
			float3 w = float3(cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta));
			float nu = dot(s, w);
			float dw = dtheta * dphi * sin(theta);
			if (order.x == 2)
			{
				// reintroducing Rayleigh and Mie in first iteration
				float pr1 = phaseFunctionR(nu);
				float pm1 = phaseFunctionM(nu);
				float3 ray1 = getDeltaSR(alt, w.z, szAngle, nu).rgb;
				float3 mie1 = getDeltaSM(alt, w.z, szAngle, nu).rgb;
				result += (ray1*pr1 + mie1*pm1)*w.z*dw;
			}
			else
			{
				result += getDeltaSR(alt, w.z, szAngle, nu).rgb * w.z* dw;
			}
		}
	}

	deltaE[DTid.xy] = float4(result, 1.0);
}