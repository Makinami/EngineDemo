Texture2D<float3> colour : register(t0);
Texture2D depth : register(t1);

Texture3D<float4> inscatter : register(t4);
Texture2D<float3> transmittance : register(t5);
Texture2D<float4> deltaE : register(t6);

#define USE_TRANSMITTANCE
#define USE_IRRADIANCE
#define USE_INSCATTER

#include "..\\Sky2\\Common.hlsli"

struct VertexOut
{
	float4 PosH : SV_POSITION;
	float3 Ray : TEXCOORD;
};

cbuffer cbPerFramePS : register(b0)
{
	float3 bCameraPos;
	float pad1;
	float3 bSunDir;
	float pad2;
	float4 gProj;
};

/*
	GLOBAL VARIABLES
*/
static float Zview = 0.0f;
static float3 surfacePos = 0.0f.xxx;

// calculate intersection with atmosphere
bool intersectAtmosphere(in float3 viewDir, out float offset, out float maxPathLength)
{
	offset = 0.0f;
	maxPathLength = 0.0f;

	float3 toCenter = -bCameraPos;
	float toCenter2 = dot(toCenter, toCenter);
	float projToView = dot(toCenter, viewDir);

	// adjusted atmosphere radius
	float R = topR - EPSILON_ATMOSPHERE;
	float R2 = R*R;

	if (toCenter2 <= R2)
	{
		// camera inside the atmosphere
		float projToRadius = toCenter2 - (projToView*projToView);
		float halfIntersection = sqrt(R2 - projToRadius);
		maxPathLength = projToView + halfIntersection;

		return true;
	}
	else if (projToView >= 0)
	{
		// camera outside
		float projToRadius = toCenter2 - (projToView*projToView);
		
		if (projToRadius <= R2)
		{
			// looking at atmosphere
			float halfIntersection = sqrt(R2 - projToRadius);
			offset = projToView - halfIntersection;
			maxPathLength = 2.0f * halfIntersection;

			return true;
		}
	}

	return false;
}

float3 GetInscatter(in float3 viewDir, out float3 attenuation, inout float irradianceFactor)
{
	float3 inscatteredLight = 0.0f.xxx;

	attenuation = 0.0.xxxx;
	float offset;
	float maxPathLength;

	if (intersectAtmosphere(viewDir, offset, maxPathLength) && Zview > offset)
	{
		// offset camera
		float3 startPos = bCameraPos + offset * viewDir;
		float startPosR = length(startPos);
		float pathLength = Zview - offset;
		// now startPos definitely inside atmosphere

		float vzAngleStart = dot(startPos, viewDir) / startPosR;
		float vsAngle = dot(viewDir, bSunDir);
		float szAngleStart = dot(startPos, bSunDir) / startPosR;

		float4 inscatter = getInscatter(startPosR, vzAngleStart, szAngleStart, vsAngle);

		//return inscatter;
		float surfacePosR = length(surfacePos);
		float szAngleEnd = dot(surfacePos, bSunDir) / surfacePosR;
		//return (maxPathLength - pathLength).xxx;
		// if surface if inside the atmosphere
		if (pathLength < maxPathLength)
		{
			// reduce inscatter light to start-surface path
			attenuation = analyticTransmittance(startPosR, vzAngleStart, pathLength);

			float vzAngleEnd = dot(surfacePos, viewDir) / surfacePosR;
			float4 inscatterAtSurface = getInscatter(surfacePosR, vzAngleEnd, szAngleEnd, vsAngle);

			inscatter = max(inscatter - attenuation.rgbr*inscatterAtSurface, 0.0f);
			irradianceFactor = 1.0f;
		}
		else
		{
			// extinction factor for infinite ray
			attenuation = analyticTransmittance(startPosR, vzAngleStart, maxPathLength);
		}

		// avoids imprecision problems near horizon by interpolating between two points above and below horizon
		float vzHorizon = -sqrt(1.0f - (groundR / startPosR)*(groundR / startPosR));
		if (abs(vzAngleStart - vzHorizon) < EPSILON_INSCATTER)
		{
			float vzAngle = vzHorizon - EPSILON_INSCATTER;
			float samplePosR = sqrt(startPosR*startPosR + pathLength*pathLength + 2.0*startPosR*pathLength*vzAngle);

			// TODO: I don't get the next line.
			float vzAngleSample = (startPosR*vzAngle + pathLength) / samplePosR;
			float4 inScatter0 = getInscatter(startPosR, vzAngle, szAngleStart, vsAngle);
			float4 inScatter1 = getInscatter(samplePosR, vzAngleSample, szAngleEnd, vsAngle);
			float4 inScatterA = max(inScatter0 - attenuation.rgbr*inScatter1, 0.0f);

			vzAngle = vzHorizon + EPSILON_INSCATTER;
			samplePosR = sqrt(startPosR*startPosR + pathLength*pathLength + 2.0*startPosR*pathLength*vzAngle);
		
			// TODO: I don't get the next line.
			vzAngleSample = (startPosR*vzAngle + pathLength) / samplePosR;
			inScatter0 = getInscatter(startPosR, vzAngle, szAngleStart, vsAngle);
			inScatter1 = getInscatter(samplePosR, vzAngleSample, szAngleEnd, vsAngle);
			float4 inScatterB = max(inScatter0 - attenuation.rgbr*inScatter1, 0.0f);

			float t = ((vzAngleStart - vzHorizon) + EPSILON_INSCATTER) / (2.0f * EPSILON_INSCATTER);

			inscatter = lerp(inScatterA, inScatterB, t);
		}

		// avoid imprecision problems in Mie scattering when sun is below horizon
		inscatter.w *= smoothstep(0.00f, 0.02f, szAngleStart);
		float phaseR = phaseFunctionR(vsAngle);
		float phaseM = phaseFunctionM(vsAngle);
		inscatteredLight = max(inscatter.rgb*phaseR + getMie(inscatter)*phaseM, 0.0f);

	}

	return inscatteredLight * 100.0;
}

float4 main( VertexOut vout ) : SV_TARGET
{
	//return float4(depth[vout.PosH.xy].ggg, 1.0);
	float Zndc = depth[vout.PosH.xy];
	Zview = Zndc == 0.0f ? 1.0f/0.0f : -(Zndc*gProj.w - gProj.z) / (Zndc*gProj.y - gProj.x) / 1000.0f;
	surfacePos = bCameraPos + Zview * vout.Ray;

	float3 attenuation = 0.0f.xxx;
	float irradianceFactor = 0.0f;

	float3 inscatterLight = GetInscatter(normalize(vout.Ray), attenuation, irradianceFactor);

	return float4(colour[vout.PosH.xy]*attenuation+inscatterLight, 1.0);
	//return float4((Zview > 100.0 ? 1.0 : 0.0).xxx, 1.0);
}

/*
Texture3D<float4> inscatterTex : register(t0);
Texture2D<float3> transmittance : register(t1);
Texture2D<float4> deltaE : register(t2);

Texture2D depth : register(t3);

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

cbuffer cbPerFramePS : register(b0)
{
	float3 bCameraPos;
	float pad1;
	float3 bSunDir;
	float pad2;
	float4 gProj;
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
		float2 coords = float2(atan(n.y / n.x), acos(n.z))*float2(0.5, 1.0) / PI + float2(0.5, 0.0);
		float4 reflectance = float4(0.1, 0.1, 0.1, 1.0); // get reflecatnce (texture or etc.)
		if (r0 > Rg + 0.01)
			reflectance = float4(0.4, 0.4, 0.4, 0.0);

		// direct sun light (radiance) reaching x0
		float muS = dot(n, s);
		float3 sunLight = getTransmittanceWithShadow(r0, muS);

		// precomputed sky light (irradiance) (=E[L*]) at x0
		float3 groundSkyLight = getIrradiance(r0, muS).rgb;

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

float4 main(VertexOut vout) : SV_TARGET
{
	float Zndc = depth[vout.PosH.xy];
	float Zview = -(Zndc*gProj.w - gProj.z) / (Zndc*gProj.y - gProj.x);

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

	return float4((sunColour + groundColour + inscatterColour), 1.0); // Eq(16)
}
*/