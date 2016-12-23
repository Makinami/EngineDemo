Texture2D<float3> transmittance : register(t0);

RWTexture3D<float4> deltaS[2] : register(u0);

#define USE_TRANSMITTANCE

#include "Common.hlsli"

void integrand(in float alt, in float vzAngle, in float szAngle, in float vsAngle, in float dist, out float3 ray, out float3 mie)
{
	ray = 0.0.xxx;
	mie = 0.0.xxx;

	float alti = sqrt(alt*alt + dist*dist + 2.0*alt*dist*vzAngle);
	float szAnglei = (vsAngle*dist + szAngle*alt) / alti;
	alti = max(alti, groundR);

	// if angle between zenith and sun is smaller than angle to horizon
	// return ray and mie scattering
	if (szAnglei >= -sqrt(1.0 - (groundR*groundR) / (alti*alti)))
	{
		float3 transi = getTransmittance(alt, vzAngle, dist) * getTransmittance(alti, szAnglei);

		ray = exp(-(alti - groundR) / HR)*transi;
		mie = exp(-(alti - groundR) / HM)*transi;
	}
}

/*void integrands(float r, float mu, float muS, float nu, float t, out float3 ray, out float3 mie)
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
}*/

void inscatter(in float alt, in float vzAngle, in float szAngle, in float vsAngle, out float3 ray, out float3 mie)
{
	ray = 0.0;
	mie = 0.0;

	float dx = intersectAtmosphereBoundry(alt, vzAngle) / float(INSCATTER_INTEGRAL_SAMPLES);
	float xi = 0.0;
	float3 rayi;
	float3 miei;

	integrand(alt, vzAngle, szAngle, vsAngle, 0.0, rayi, miei);

	for (int i = 1; i < INSCATTER_INTEGRAL_SAMPLES; ++i)
	{
		float xj = float(i)*dx;
		float3 rayj;
		float3 miej;
		integrand(alt, vzAngle, szAngle, vsAngle, xj, rayj, miej);
		ray += (rayj + rayi) / 2.0 * dx;
		mie += (miej + miei) / 2.0 * dx;
		rayi = rayj;
		miei = miej;
	}

	ray *= betaR;
	mie *= betaMSca;
}

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	float alt = DTid.z / (RES_ALT - 1.0);
	alt = sqrt(groundR*groundR + alt*alt*(topR*topR - groundR*groundR)) + (DTid.z == 0 ? 0.01 : (DTid.z == RES_ALT - 1 ? -0.01 : 0.0));
	float4 dhdH = float4(topR - alt, sqrt(alt*alt - groundR*groundR) + sqrt(topR*topR - groundR*groundR), alt - groundR, sqrt(alt*alt - groundR*groundR));

	float3 ray;
	float3 mie;
	float vzAngle, szAngle, vsAngle;

	getVzSzVsAngles(float2(DTid.xy), alt, dhdH, vzAngle, szAngle, vsAngle);
	inscatter(alt, vzAngle, szAngle, vsAngle, ray, mie);

	deltaS[0][DTid] = float4(ray, 0.0);
	deltaS[1][DTid] = float4(mie, 0.0);
}