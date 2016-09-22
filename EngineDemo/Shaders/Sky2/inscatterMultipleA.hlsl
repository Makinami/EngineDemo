Texture2D<float4> transmittance : register(t0);

Texture2D<float4> deltaE : register(t1);
Texture3D<float4> deltaSR : register(t2);
Texture3D<float4> deltaSM : register(t3);

RWTexture3D<float4> deltaJ : register(u0);

SamplerState samBilinearClamp : register(s0);

#define USE_TRANSMITTANCE
#define USE_IRRADIANCE
#define USE_DELTAS

#include "Common.hlsli"

cbuffer Order : register(b0)
{
	int4 order;
};

static const float dphi = PI / float(INSCATTER_SPHERICAL_INTEGRAL_SAMPLES);
static const float dtheta = PI / float(INSCATTER_SPHERICAL_INTEGRAL_SAMPLES);

void inscatter(float alt, float vzAngle, float szAngle, float vsAngle, out float3 raymie)
{
	alt = clamp(alt, groundR, topR);
	vzAngle = clamp(vzAngle, -1.0, 1.0);
	szAngle = clamp(szAngle, -1.0, 1.0);
	float var = sqrt(1.0 - vzAngle*vzAngle)*sqrt(1.0 - szAngle*szAngle);
	vsAngle = clamp(vsAngle, vzAngle*szAngle - var, vzAngle*szAngle + var);

	float cthetamin = -sqrt(1.0 - (groundR*groundR) / (alt*alt));

	float3 v = float3(sqrt(1.0 - vzAngle*vzAngle), 0.0, vzAngle);
	float sx = (v.x == 0.0) ? 0.0 : (vsAngle - szAngle*vzAngle) / v.x;
	float3 s = float3(sx, sqrt(max(0.0, 1.0 - sx*sx - szAngle*szAngle)), szAngle);

	// integral over 4.PI around x with two nested loops over w directions (theta, pi) -- Eq (7)
	for (int itheta = 0; itheta < INSCATTER_SPHERICAL_INTEGRAL_SAMPLES; ++itheta)
	{
		float theta = (float(itheta) + 0.5)*dtheta;
		float ctheta = cos(theta);

		float greflectance = 0.0;
		float dground = 0.0;
		float3 gtrans = float3(0.0, 0.0, 0.0);

		if (ctheta < cthetamin) // ground visible in direction w
		{
			// compute transparrency gtrans beetween x and ground
			greflectance = AVERAGE_GROUND_REFLECTANCE / PI;
			dground = -alt * ctheta - sqrt(alt*alt*(ctheta*ctheta - 1.0) + groundR*groundR);
			gtrans = getTransmittance(groundR, -(alt*ctheta + dground) / groundR, dground);
		}

		for (int iphi = 0; iphi < 2 * INSCATTER_SPHERICAL_INTEGRAL_SAMPLES; ++iphi)
		{
			float phi = (float(iphi) + 0.5)*dphi;
			float dw = dtheta * dphi * sin(theta);
			float3 w = float3(cos(phi)*sin(theta), sin(phi)*sin(theta), ctheta);

			float nu1 = dot(s, w);
			float nu2 = dot(v, w);
			float pr2 = phaseFunctionR(nu2);
			float pm2 = phaseFunctionM(nu2);

			// compute irradiance received at ground in direction w (if ground visible) = deltaE
			float3 gnormal = (float3(0.0, 0.0, alt) + dground*w) / groundR;
			float3 girradiance = getIrradiance(groundR, dot(gnormal, s));

			float3 raymie1; // light arriving at x from direction w

							// first term = light reflected from the ground and attenuated before reaching x, = T.alpha/PI.deltaE
			raymie1 = greflectance * girradiance * gtrans;

			// second termp = inscattered light, = deltaS
			if (order.x == 2)
			{
				// first iteration is special because Rayleigh and Mie were stored separately,
				// without the phase function factors; they must be reintroduces here
				float pr1 = phaseFunctionR(nu1);
				float pm1 = phaseFunctionM(nu1);
				float3 ray1 = getDeltaSR(alt, w.z, szAngle, nu1);
				float3 mie1 = getDeltaSM(alt, w.z, szAngle, nu1);
				raymie1 += ray1 * pr1 + mie1 * pm1;
			}
			else
			{
				raymie1 += getDeltaSR(alt, w.z, szAngle, nu1);
			}

			// light coming from direction w (raymie1) * SUM(scattering coefficient * phaseFunction)
			// Eq (7)
			raymie += raymie1 * (betaR * exp(-(alt - groundR) / HR) * pr2 + betaMSca * exp(-(alt - groundR) / HM) * pm2)*dw;
		}
	}
}

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	float alt = DTid.z / (RES_ALT - 1.0);
	alt = alt * alt;
	alt = sqrt(groundR*groundR + alt*(topR*topR - groundR*groundR)) + (DTid.z == 0 ? 0.01 : (DTid.z == RES_ALT - 1 ? -0.01 : 0.0));
	float4 dhdH = float4(topR - alt, sqrt(alt*alt - groundR*groundR) + sqrt(topR*topR - groundR*groundR), alt - groundR, sqrt(alt*alt - groundR*groundR));

	float3 raymie = 0.0.xxx;
	float vzAngle, szAngle, vsAngle;
	getVzSzVsAngles(DTid.xy, alt, dhdH, vzAngle, szAngle, vsAngle);
	inscatter(alt, vzAngle, szAngle, vsAngle, raymie);
	deltaJ[DTid] = float4(raymie, 0.0);
}