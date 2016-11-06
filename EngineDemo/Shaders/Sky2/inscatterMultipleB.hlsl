Texture2D<float3> transmittance : register(t0);
Texture3D<float4> deltaJ : register(t1);

RWTexture3D<float4> deltaSR : register(u0);

#define USE_TRANSMITTANCE
#define USE_DELTAJ

#include "Common.hlsli"

float3 integrand(float alt, float vzAngle, float szAngle, float vsAngle, float t)
{
	float alt_i = sqrt(alt*alt + t*t + 2.0*alt*vzAngle*t);
	float vzAngle_i = (alt*vzAngle + t) / alt_i;
	float szAngle_i = (vsAngle*t + szAngle*alt) / alt_i;
	return getDeltaJ(alt_i, vzAngle_i, szAngle_i, vsAngle).rgb * getTransmittance(alt, vzAngle, t);
}

float3 inscatter(float alt, float vzAngle, float szAngle, float vsAngle, out float3 raymie)
{
	float dx = intersectAtmosphereBoundry(alt, vzAngle) / float(INSCATTER_INTEGRAL_SAMPLES);
	float x_i = 0.0;
	float3 raymie_i = integrand(alt, vzAngle, szAngle, vsAngle, 0.0);
	for (int i = 1; i <= INSCATTER_INTEGRAL_SAMPLES; ++i)
	{
		float x_j = float(i) * dx;
		float3 raymie_j = integrand(alt, vzAngle, szAngle, vsAngle, x_j);
		raymie += (raymie_i + raymie_j) / 2.0 *dx;
		x_i = x_j;
		raymie_i = raymie_j;
	}

	return raymie;
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
	deltaSR[DTid] = float4(raymie, 1.0);
}