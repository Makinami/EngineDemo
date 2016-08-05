Texture2D<float3> transmittance : register(t1);
Texture2D<float4> Noise : register(t2);

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

cbuffer MatrixBuffer
{
	matrix gViewInverse;
	matrix gProjInverse;
	float3 gCameraPos;
	float time;
	float3 gSunDir;
	float pad2;
};

struct PixelInputType
{
	float4 PosH : SV_POSITION;
	float3 Ray : TEXCOORD;
};

float noise(in float3 x)
{
	float3 p = floor(x);
	float3 f = frac(x);
	f = f*f*(3.0 - 2.0*f);
	float2 uv = (p.xy + float2(37.0, 17.0)*p.z) + f.xy;
	float2 rg = Noise.Sample(samTransmittance, (uv + 0.5) / 256.0).yx;
	return -1.0 + 2.0*lerp(rg.x, rg.y, f.z);
}

float map5(in float3 p)
{
	float3 q = p - float3(0.0, 0.1, 1.0) * time;
	float f;
	f  = 0.50000*noise(q); q *= 2.02;
	f += 0.25000*noise(q); q *= 2.03;
	f += 0.12500*noise(q); q *= 2.01;
	f += 0.06250*noise(q); q *= 2.02;
	f += 0.03125*noise(q);
	return clamp(1.5 - p.y - 2.0 + 1.75*f, 0.0, 1.0);
}

float4 integrate(in float4 sum, in float dif, in float den, in float bgcol, in float t)
{
	// lighting
	float3 lin = float3(0.65, 0.7, 0.75)*1.4 + float3(1.0, 0.6, 0.3)*dif;
	float4 col = float4(lerp(float3(1.0, 0.95, 0.8), float3(0.25, 0.3, 0.35), den), den);
	col.rgb *= lin;
	col.rgb = lerp(col.rgb, bgcol, 1.0 - exp(-0.003*t*t));
	// front to back blending
	col.a *= 0.4;
	col.rgb *= col.a;
	return sum + col*(1.0 - sum.a);
}

#define MARCH(STEPS,MAPLOD) for(int i=0; i<STEPS; ++i) { float3 pos = origin + t*dir; if(pos.y <-3.0 || pos.y > 2.0 || sum.a > 0.99) break; float den = MAPLOD(pos); if(den>0.01) { float dif = clamp((den - MAPLOD(pos+0.3*gSunDir)/0.6), 0.0, 1.0); sum = integrate(sum, dif, den, bgcol, t); } t += max(0.05, 0.02*t); }

float4 raymarch(in float3 origin, in float3 dir, in float3 bgcol)
{
	float4 sum = float4(0.0, 0.0, 0.0, 0.0);

	float t = 0.0;

	MARCH(30, map5);
	//MARCH(30, map4);
	//MARCH(30, map3);
	//MARCH(30, map2);

	return clamp(sum, 0.0, 1.0);
}

float4 main(PixelInputType pin) : SV_TARGET
{
	// backgroud sky
	float sun = clamp(dot(gSunDir,pin.Ray), 0.0, 1.0);
	float3 col = float3(0.6,0.71,0.75) - pin.Ray.y*0.2*float3(1.0,0.5,1.0) + 0.15*0.5;
	//col += 0.2*float3(1.0, .6, 0.1)*pow(sun, 8.0);

	// clouds
	float4 res = raymarch(gCameraPos, pin.Ray, col);
	col = col*(1.0 - res.w) + res.xyz;

	// sun glare    
	//col += 0.2*float3(1.0, 0.4, 0.2)*pow(sun, 3.0);

	return float4(col, 1.0f);
}