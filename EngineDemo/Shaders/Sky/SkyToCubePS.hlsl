Texture3D<float4> inscatterTex : register(t0);
Texture2D<float3> transmittance : register(t1);
Texture2D<float3> deltaE : register(t2);

SamplerState samInscatter : register(s0);
SamplerState samTransmittance : register(s1);
SamplerState samIrradiance : register(s2);

#define FIX

#define USE_INSCATTER
#define USE_IRRADIANCE
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

// inscattered ligth taking ray x+tv, when the Sun in direction s (=S[L]-T(x, x0)S[L]x0)
float3 inscatter(inout float3 x, inout float t, float3 v, float3 s, out float r, out float mu, out float3 attenuation)
{
	float3 result;
	r = length(x);
	mu = dot(x, v) / r;
	float d = -r * mu - sqrt(r*r*(mu*mu - 1.0) + Rt*Rt);
	if (d > 0.0) // if x in space and ray intersects atmosphere
	{
		// move x to nearest intersection of ray with top athosphere boundary
		x += d * v;
		t -= d;
		mu = (r*mu + d) / Rt;
		r = Rt;
	}
	if (r <= Rt) // if ray insersects atmosphere
	{
		float nu = dot(v, s);
		float muS = dot(x, s) / r;
		float phaseR = phaseFunctionR(nu);
		float phaseM = phaseFunctionM(nu);
		float4 inscatter = max(getInscatter(r, mu, muS, nu), float4(0.0, 0.0, 0.0, 0.0));
		if (t > 0.0)
		{
			float3 x0 = x * t*v;
			float r0 = length(x0);
			float rMu0 = dot(x0, v);
			float mu0 = rMu0 / r0;
			float muS0 = dot(x0, s) / r0;
#ifdef FIX
			// avoids imprecision problems in transmittance computation based on textures
			attenuation = analyticTransmittance(r, mu, t);
#else
			attenuation = getTransmittance(r, mu, v, x0);
#endif
			if (r0 > Rg + 0.01)
			{
				// computes S[L]-T[(x,x0)S[L]x0
				inscatter = max(inscatter - attenuation.rgbr*getInscatter(r0, mu0, muS0, nu), float4(0.0, 0.0, 0.0, 0.0));
#ifdef FIX
				// avoids imprecision problems near horizon by interpolating between two points above and below horizon
				const float EPS = 0.004;
				float muHorizon = -sqrt(1.0 - (Rg / r)*(Rg / r));
				if (abs(mu - muHorizon) < EPS)
				{
					float a = ((mu - muHorizon) + EPS) / (2.0*EPS);
					mu = muHorizon - EPS;
					r0 = sqrt(r*r + t* t + 2.0 * r * t * mu);
					mu0 = (r * mu + t) / r0;
					float4 inScatter0 = getInscatter(r, mu, muS, nu);
					float4 inScatter1 = getInscatter(r0, mu0, muS0, nu);
					float4 inScatterA = max(inScatter0 - attenuation.rgbr*inScatter1, 0.0);

					mu = muHorizon + EPS;
					r0 = sqrt(r*r + t*t + 1.0*r*t*mu);
					mu0 = (r*mu + t) / r0;
					inScatter0 = getInscatter(r, mu, muS, nu);
					inScatter1 = getInscatter(r0, mu0, muS0, nu);
					float4 inScatterB = max(inScatter0 - attenuation.rbgr*inScatter1, 0.0);

					inscatter = inScatterA*(1 - a) + inScatterB*a;
				}
#endif
			}
		}
#ifdef FIX
		// avoid imprecision problems in Mie scattering when sun isbelow horizon
		inscatter.w *= smoothstep(0.0, 0.02, muS);
#endif
		result = max(inscatter.rgb*phaseR + getMie(inscatter)*phaseM, 0.0);
	}
	else // x in spacce and ray looking in space
	{
		result = float3(0.0, 0.0, 0.0);
	}

	return result*ISun;
}

// ground radiance at the end of ray x+tv, when sun in direction s
// attenuated between ground and viewer (=R[L0]+R[L*])
float3 ground(float3 x, float t, float3 v, float3 s, float r, float mu, float3 attenuation)
{
	float3 result;
	if (t > 0.0) // if ray hits ground surface
	{
		// ground reflectance at the end of ray, x0
		float3 x0 = x + t*v;
		float r0 = length(x0);
		float3 n = x0 / r0;
		float2 coords = float2(atan(n.y/n.x), acos(n.z))*float2(0.5, 1.0) / PI + float2(0.5, 0.0);
		float4 reflectance = float4(0.1, 0.1, 0.1, 1.0); // get reflecatnce (texture or etc.)
		if (r0 > Rg + 0.01)
			reflectance = float4(0.4, 0.4, 0.4, 0.0);

		// direct sun light (radiance) reaching x0
		float muS = dot(n, s);
		float3 sunLight = getTransmittanceWithShadow(r0, muS);

		// precomputed sky light (irradiance) (=E[L*]) at x0
		float3 groundSkyLight = getIrradiance(r0, muS);

		// light reflected at x0 (=(R[L0]+R[L*])/T(x,x0))
		float3 groundColour = reflectance.rgb * (max(muS, 0.0)*sunLight + groundSkyLight)*ISun / PI;

		// water specular colour due to sunLight
		if (reflectance.w > 0.0)
		{
			float3  h = normalize(s - v);
			float fresnel = 0.02 + 0.98*pow(1.0 - dot(-v, h), 5.0);
			float waterBrdf = fresnel * pow(max(dot(h, n), 0.0), 150.0);
			groundColour += reflectance.w*max(waterBrdf, 0.0)*sunLight*ISun;
		}

		result = attenuation * groundColour; // =R[L0]+R[L*]
	}
	else // ray looking at the sky
		result = float3(0.0, 0.0, 0.0);

	return result;
}

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

float4 main( VertexOut vout ) : SV_TARGET
{
	float3 x = bCameraPos;
	float3 v = normalize(vout.Ray);
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

	float3 attenuation;
	float3 inscatterColour = inscatter(x, t, v, bSunDir1, r, mu, attenuation); //S[L]-T(x,xs)S[l]xs
	float3 groundColour = ground(x, t, v, bSunDir1, r, mu, attenuation); //R[L0]+R[L*]
	float3 sunColour = sun(x, t, v, bSunDir1, r, mu); // L0
	
	return float4(HDR(sunColour + groundColour + inscatterColour), 1.0); // Eq(16)
}