Texture2D<float3> transmittance : register(t0);
Texture3D<float4> deltaJ : register(t1);

RWTexture3D<float4> deltaSR : register(u0);

SamplerState samTransmittance : register(s0);
SamplerState samDeltaJ : register(s1);

#define USE_TRANSMITTANCE
#define USE_DELTAJ

#pragma warning(disable:3568)
#include <resolutions.h>
#pragma warning(default:3568)

#include <common.hlsli>

float3 integrand(float r, float mu, float muS, float nu, float t)
{
	float ri = sqrt(r*r + t*t + 2.0*r*mu*t);
	float mui = (r*mu + t) / ri;
	float muSi = (nu*t + muS*r) / ri;
	return getDeltaJ(ri, mui, muSi, nu).rgb * getTransmittance(r, mu, t);
}

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	float r = DTid.z / (RES_R - 1.0f);
	r = r * r;
	r = sqrt(Rg*Rg + r*(Rt*Rt - Rg*Rg)) + (DTid.z == 0 ? 0.01 : (DTid.z == RES_R - 1 ? -0.001 : 0.0));
	float4 dhdH = float4(Rt - r, sqrt(r*r - Rg*Rg) + sqrt(Rt*Rt - Rg*Rg), r - Rg, sqrt(r*r - Rg*Rg));

	float mu, muS, nu;
	getMuMuSNu(DTid.xy, r, dhdH, mu, muS, nu);

	float3 raymie = float3(0.0, 0.0, 0.0);
	float dx = limit(r, mu) / float(INSCATTER_INTEGRAL_SAMPLES);
	float xi = 0.0;
	float3 raymiei = integrand(r, mu, muS, nu, 0);
	for (int i = 1; i <= INSCATTER_INTEGRAL_SAMPLES; ++i)
	{
		float xj = float(i)*dx;
		float3 raymiej = integrand(r, mu, muS, nu, xj);
		raymie += (raymiei + raymiej) / 2.0 * dx;
		xi = xj;
		raymiei = raymiej;
	}

	deltaSR[DTid] = float4(raymie, 0.0);
}