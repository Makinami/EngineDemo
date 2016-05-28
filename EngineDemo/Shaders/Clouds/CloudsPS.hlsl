Texture2D<float3> transmittance : register(t1);
TextureCube gCubeMap : register(t3);
Texture3D<float4> cloudsGeneral : register(t4);
Texture3D<float4> cloudsDetail : register(t5);
Texture2D<float4> cloudsCurl : register(t6);
Texture2D<float4> cloudsType : register(t7);
Texture2D<float4> weatherPar : register(t8);

static const float Rl = 6361.5;
static const float Rh = 6364.0;

SamplerState samTransmittance : register(s1);

SamplerState samTrilinearSam : register(s3);

#define FIX

#define USE_TRANSMITTANCE

#pragma warning(disable:3568)
#include <..//Sky//resolutions.h>
#pragma warning(default:3568)

#include <..\Sky\common.hlsli>

static const float ISun = 100.0;
static const float G_SCATTERING = 0.9f;

cbuffer cbPerFramePS
{
	float3 bCameraPos;
	float bExposure;
	float3 bSunDir;
	float2 pad;
};

struct PixelInputType
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

float ComputeScattering(float lightDotView)
{
	float result = 1.0f - G_SCATTERING * G_SCATTERING;
	result /= (4.0f * PI * pow(1.0f + G_SCATTERING * G_SCATTERING - (2.0f * G_SCATTERING) * lightDotView, 1.5f));
	return result;
}

float sampleCheap(float3 x)
{
	float4 clouds = cloudsGeneral.SampleLevel(samTrilinearSam, float3(x.x/4, x.y, x.z/4), 0);
	float4 weather = weatherPar.SampleLevel(samTrilinearSam, (x.xz + float2(70.0f, 70.0f))/140.0f, 0);
	float4 type = cloudsType.SampleLevel(samTrilinearSam, float2(weather.z, 1-x.y), 0);

	float4 base = clouds * type.x;
	float4 cutoff = step(1.0f - weather.x, base);

	float4 alpha = cutoff * base * weather.x * (0.625f * x.y + 0.375f);

	return alpha.x;
}

float sampleExpensive(float3 x)
{
	float4 clouds = cloudsGeneral.SampleLevel(samTrilinearSam, float3(x.x / 4, x.y, x.z / 4), 0);
	float4 weather = weatherPar.SampleLevel(samTrilinearSam, (x.xz + float2(70.0f, 70.0f)) / 140.0f, 0);
	float4 type = cloudsType.SampleLevel(samTrilinearSam, float2(weather.z, 1 - x.y), 0);

	float4 base = clouds * type.x;
	float4 cutoff = step(1.0f - weather.x, base);

	float4 cloudsBase = cutoff * base * weather.x * (0.625f * x.y + 0.375f);

	float4 cloudsDet = (1.0f + cloudsDetail.SampleLevel(samTrilinearSam, float3(x.x*4.0f, x.y*4.0f, x.z*4.0f), 0))/2.0f;
	float4 curl = cloudsCurl.SampleLevel(samTrilinearSam, float2(x.xz), 0);

	float4 cloudsDif = cloudsDet*curl;

	float4 alpha = clamp(cloudsBase - cloudsDif, 0.0f, 2.0f);

	return alpha.x;
}

float2 rayMarchAlpha(float3 y0, float3 y1, float mu)
{
	int steps = -64 * mu + 128;
	float2 transmittance = float2(1.0f, 0.0f);
	
	uint zero = 0;
	y0.y = 0.0f; y1.y = 1.0f;
	float3 v = (y1 - y0) / 64.0f;
	float step = 1.0f;
	float da = 0.0f;
	float i = 0.0f;

	for (i = 0.0f; i < 64; i += step)
	{
		da = 0.0f;
		if (transmittance.x < 0.7)
		{
			da = sampleCheap(y0 + i*v);
			transmittance.x *= exp(-step*da);
			transmittance.y += exp(-step*da)*(1 - exp(-i * 2));
			step = 1.0f;
		}
		else
		{
			if (step > 0.5f)
			{
				da = sampleCheap(y0 + i*v);
				if (da > 0.0f)
				{
					i = max(i - step, 0.0f);
					step = 0.1f;
				}
			}
			else if (zero < 3)
			{
				da = sampleCheap(y0 + i*v);
				if (da < 1e-9) ++zero;
				else
				{
					zero = 0;
					transmittance.x *= exp(-step*da);
					transmittance.y += exp(-step*da)*(1 - exp(-i * 2));
				}
			}
			else
			{
				da = sampleCheap(y0 + i*v);
				transmittance.x *= exp(-step*da);
				transmittance.y += exp(-step*da)*(1 - exp(-i * 2));
				step = 1.0f;
			}
		}
		if (transmittance.x <= 0.01f) break;
	}

	return transmittance;
}

float4 main(PixelInputType pin) : SV_TARGET
{
	float3 v = normalize(pin.Ray);
	float3 x = bCameraPos;
	float3 bSunDir1 = normalize(bSunDir);

	float r = length(x);
	float mu = dot(x, v) / r;
	float t0 = -r * mu + sqrt(r * r * (mu * mu - 1.0f) + Rl * Rl);
	float t1 = -r * mu + sqrt(r * r * (mu * mu - 1.0f) + Rh * Rh);
	float tg = -r * mu + sqrt(r * r * (mu * mu - 1.0f) + Rg * Rg);

	clip(isnan(tg) || tg >= 0 ? -1 : 1);

	float3 g = x - float3(0.0, 0.0, Rg + 10.0);
	float a = v.x * v.x + v.y * v.y - v.z * v.z;
	float b = 2.0 * (g.x * v.x + g.y * v.y - g.z * v.z);
	float c = g.x *g.x + g.y * g.y - g.z * g.z;
	float d = -(b + sqrt(b * b - 4.0 * a * c)) / (2.0 * a);
	bool cone = d > 0.0 && abs(x.z + d * v.z - Rg) <= 10.0;

	if (tg > 0.0)
	{
		if (cone && d < tg)
			tg = d;
	}
	else if (cone)
		tg = d;

	float3 sunColour = sun(x, tg, bSunDir1, bSunDir1, r, dot(bSunDir1, x)); // L0

	float3 y0 = x + t0*v;
	float3 y1 = x + t1*v;

	float2 alpha = rayMarchAlpha(y0, y1, mu);

	return float4(sunColour*alpha.y*ComputeScattering(dot(bSunDir1, v)), 1.0f-alpha.x);
}