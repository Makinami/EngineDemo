Texture2D<float3> transmittance : register(t0);

RWTexture3D<float3> deltaSR : register(u0);
RWTexture3D<float3> deltaSM : register(u1);

SamplerState samTransmittance : register(s0);

#define USE_TRANSMITTANCE

#pragma warning(disable:3568)
#include <resolutions.h>
#pragma warning(default:3568)

#include <common.hlsli>

void integrand(float r, float mu, float muS, float nu, float t, out float3 ray, out float3 mie)
{
	ray = float3(0.0f, 0.0f, 0.0f);
	mie = float3(0.0f, 0.0f, 0.0f);

	float ri = sqrt(r*r + t*t + 2.0*r*mu*t);
	float muSi = (nu*t + muS*r) / ri;
	ri = max(Rg, ri);
	if (muSi >= -sqrt(1.0 - Rg*Rg / (ri*ri)))
	{
		float3 ti = getTransmittance(r, mu, t) * getTransmittance(ri, muSi);

		ray = exp(-(ri - Rg) / HR)*ti;
		mie = exp(-(ri - Rg) / HM)*ti;
	}
}

void inscatter(float r, float mu, float muS, float nu, out float3 ray, out float3 mie)
{
	ray = float3(0.0, 0.0f, 0.0f);
	mie - float3(0.0f, 0.0f, 0.0f);

	float dx = limit(r, mu) / float(INSCATTER_INTEGRAL_SAMPLES);
	float xi = 0.0;
	float3 rayi;
	float3 miei;

	integrand(r, mu, muS, nu, 0.0, rayi, miei);

	for (int i = 1; i < INSCATTER_INTEGRAL_SAMPLES; ++i)
	{
		float xj = float(i)*dx;
		float3 rayj;
		float3 miej;
		integrand(r, mu, muS, nu, xj, rayj, miej);
		ray += (rayi + rayj) / 2.0*dx;
		mie += (miei + miej) / 2.0*dx;
		xi = xj;
		rayi = rayj;
		miei = miej;
	}
	ray *= betaR;
	mie *= betaMSca;
}

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	float r = DTid.z / (RES_R - 1.0f);
	r = r * r;
	r = sqrt(Rg*Rg + r*(Rt*Rt - Rg*Rg)) + (DTid.z == 0 ? 0.01 : (DTid.z == RES_R - 1 ? -0.01 : 0.0));
	float4 dhdH = float4(Rt - r, sqrt(r*r - Rg*Rg) + sqrt(Rt*Rt - Rg*Rg), r - Rg, sqrt(r*r - Rg*Rg));
	
	float3 ray;
	float3 mie;
	float mu, muS, nu;

	getMuMuSNu(float2(DTid.xy), r, dhdH, mu, muS, nu);
	inscatter(r, mu, muS, nu, ray, mie);

	deltaSR[DTid] = ray;
	deltaSM[DTid] = mie;
}