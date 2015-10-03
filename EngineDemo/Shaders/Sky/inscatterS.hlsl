Texture2D<float3> transmittance : register(t0);

Texture2D<float3> deltaE : register(t1);
Texture3D<float3> deltaSR : register(t2);
Texture3D<float3> deltaSM : register(t3);

RWTexture3D<float3> deltaJ : register(u0);

SamplerState samTransmittance : register(s0);
SamplerState samDeltaSR : register(s1);
SamplerState samDeltaSM : register(s2);
SamplerState samIrradiance : register(s3);

#define USE_IRRADIANCE
#define USE_TRANSMITTANCE
#define USE_DELTAS

#pragma warning(disable:3568)
#include <resolutions.h>
#pragma warning(default:3568)

#include <common.hlsli>

cbuffer Order
{
	int4 order;
};

static const float dphi = PI / float(INSCATTER_SPHERICAL_INTEGRAL_SAMPLES);
static const float dtheta = PI / float(INSCATTER_SPHERICAL_INTEGRAL_SAMPLES);

void inscatter(float r, float mu, float muS, float nu, out float3 raymie)
{
	r = clamp(r, Rg, Rt);
	mu = clamp(mu, -1.0, 1.0);
	muS = clamp(muS, -1.0, 1.0);
	float var = sqrt(1.0 - mu*mu)*sqrt(1.0 - muS*muS);
	nu = clamp(nu, muS*mu - var, muS*mu + var);

	float cthetamin = -sqrt(1.0 - (Rg / r)*(Rg / r));

	float3 v = float3(sqrt(1.0 - mu*mu), 0.0, mu);
	float sx = (v.x == 0.0) ? 0.0 : (nu - muS*mu) / v.x;
	float3 s = float3(sx, sqrt(max(0.0, 1.0 - sx*sx - muS*muS)), muS);

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
			dground = -r * ctheta - sqrt(r*r*(ctheta*ctheta - 1.0) + Rg*Rg);
			gtrans = getTransmittance(Rg, -(r*ctheta + dground) / Rg, dground);
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
			float3 gnormal = (float3(0.0, 0.0, r) + dground*w) / Rg;
			float3 girradiance = getIrradiance(Rg, dot(gnormal, s));

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
				float3 ray1 = getDeltaSR(r, w.z, muS, nu1);
				float3 mie1 = getDeltaSM(r, w.z, muS, nu1);
				raymie1 += ray1 * pr1 + mie1 * pm1;
			}
			else
			{
				raymie1 += getDeltaSR(r, w.z, muS, nu1);
			}

			// light coming from direction w (raymie1) * SUM(scattering coefficient * phaseFunction)
			// Eq (7)
			raymie += raymie1 * (betaR * exp(-(r - Rg) / HR) * pr2 + betaMSca * exp(-(r - Rg) / HM) * pm2)*dw;
		}
	}
}

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	float r = DTid.z / (RES_R - 1.0f);
	r = r * r;
	r = sqrt(Rg*Rg + r*(Rt*Rt - Rg*Rg)) + (DTid.z == 0 ? 0.01 : (DTid.z == RES_R - 1 ? -0.01 : 0.0));
	float4 dhdH = float4(Rt - r, sqrt(r*r - Rg*Rg) + sqrt(Rt*Rt - Rg*Rg), r - Rg, sqrt(r*r - Rg*Rg));
	
	float3 raymie = float3(0, 0, 0);
	float mu, muS, nu;
	getMuMuSNu(DTid.xy, r, dhdH, mu, muS, nu);
	inscatter(r, mu, muS, nu, raymie);
	deltaJ[DTid] = raymie;
}