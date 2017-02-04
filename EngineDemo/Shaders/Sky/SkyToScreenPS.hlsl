Texture2D<float3> transmittance : register(t1);
TextureCube gCubeMap : register(t3);

SamplerState samTransmittance : register(s1);
SamplerState samTrilinearSam : register(s3);

#define FIX

#define USE_TRANSMITTANCE

#pragma warning(disable:3568)
#include <resolutions.h>
#pragma warning(default:3568)

#include <common.hlsli>

static const float ISun = 100.0;

cbuffer cbPerFramePS
{
	float3 bCameraPos;
	float bExposure;
	float3 bSunDir;
	float pad;
};

struct VertexOut
{
	float4 PosH : SV_POSITION;
	float3 Ray : TEXCOORD;
};

// direct sun light for ray x+tv, when sun in direction s (=L0)
float3 sun(float3 x, float t, float3 v, float3 s, float r, float mu)
{
	if (t > 0.0)
		return float3(0.0, 0.0, 0.0);
	else
	{
		float3 transmittance = r <= Rt ? getTransmittanceWithShadow(r, mu) : float3(1.0, 1.0, 1.0); // T(x,x0)
		float isun = step(cos(PI / 180.0), dot(v, s)) * ISun; // Lsun dot(s, v);// 
		return transmittance*isun; // Eq (9)
	}
}

float3 HDR(float3 L)
{
	L = L*bExposure;
	L.r = L.r < 1.413 ? pow(L.r * 0.38317, 1.0 / 2.2) : 1.0 - exp(-L.r);
	L.g = L.g < 1.413 ? pow(L.g * 0.38317, 1.0 / 2.2) : 1.0 - exp(-L.g);
	L.b = L.b < 1.413 ? pow(L.b * 0.38317, 1.0 / 2.2) : 1.0 - exp(-L.b);
	return L;
}

float4 main(VertexOut pin) : SV_TARGET
{
	float3 x = bCameraPos;
	float3 v = normalize(pin.Ray);
	float3 bSunDir1 = normalize(bSunDir);

	float r = length(x);
	float mu = dot(x, v) / r;
	float t = -r * mu - sqrt(r * r * (mu * mu - 1.0) + Rg * Rg);

	float3 g = x - float3(0.0, 0.0, Rg + 10.0);
	float a = v.x * v.x + v.y * v.y - v.z * v.z;
	float b = 2.0 * (g.x * v.x + g.y * v.y - g.z * v.z);
	float c = g.x *g.x + g.y * g.y - g.z * g.z;
	float d = -(b + sqrt(b * b - 4.0 * a * c)) / (2.0 * a);
	bool cone = d > 0.0 && abs(x.z + d * v.z - Rg) <= 10.0;

	if (t > 0.0)
	{
		if (cone && d < t)
			t = d;
	}
	else if (cone)
		t = d;

	float3 sunColour = sun(x, t, v, bSunDir1, r, mu); // L0

	return float4((gCubeMap.Sample(samTrilinearSam, pin.Ray).xyz + sunColour), 1.0f);
}