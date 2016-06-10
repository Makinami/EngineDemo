#include "atmosphere.hlsli"

cbuffer WaterParams : register(c0)
{
	matrix screenToCamera; // screen space to camera space
	matrix cameraToWorld; // camera space to world space
	matrix worldToScreen; // world space to screen space
	float3 worldCamera; // camera position
	float normals;
	float3 worldSunDir; // sun direction in world space
	float choppy;
	float4 GRID_SIZES;
	float3 seaColour;
	float pad;
	float2 gridSize;
};

Texture2DArray<float4> gDisplacement : register(t0);
Texture3D<float4> gSlopeVariance : register(t1);
SamplerState samFFTMap : register(s2);
SamplerState samVariance : register(s3);

struct VertexOut
{
	float4 PosH : SV_POSITION;
	float3 PosW : TEXTCOORD0;
	float2 u : TEXTCOORD1;
};

// ---------------------------------------------------------------------
// REFLECTED SUN RADIANCE
// ---------------------------------------------------------------------

// assert x>0
float erfc(float x)
{
	return 2.0 * exp(-x*x) / (2.319*x + sqrt(4.0 + 1.52*x*x));
}

float Lambda(float cosTheta, float sigmaSq)
{
	float v = cosTheta / sqrt((1.0 - cosTheta * cosTheta) * (2.0 * sigmaSq));
	return max(0.0, (exp(-v*v) - v*sqrt(PI) * erfc(v)) / (2.0 * v * sqrt(PI)));
	// return (exp(-v*v)) / (2.0 * v * sqrt(PI)); // approximate, faster formula
}

// L, V, N, Tx, Ty in world space
float reflectedSunRadiance(float3 L, float3 V, float3 N, float3 Tx, float3 Ty, float2 sigmaSq)
{
	float3 H = normalize(L + V);
	float zetax = dot(H, Tx) / dot(H, N);
	float zetay = dot(H, Ty) / dot(H, N);

	float zL = dot(L, N); // cos of source zenith angle
	float zV = dot(V, N); // cos of receiver zenith angle
	float zH = dot(H, N); // cos of facet normal zenith angle
	float zH2 = zH * zH;

	float p = exp(-0.5 * (zetax * zetax / sigmaSq.x + zetay * zetay / sigmaSq.y)) / (2.0 * PI * sqrt(sigmaSq.x * sigmaSq.y));

	float tanV = atan2(dot(V, Ty), dot(V, Tx));
	float cosV2 = 1.0 / (1.0 + tanV * tanV);
	float sigmaV2 = sigmaSq.x * cosV2 + sigmaSq.y * (1.0 - cosV2);

	float tanL = atan2(dot(L, Tx), dot(L, Ty));
	float cosL2 = 1.0 / (1.0 + tanL * tanL);
	float sigmaL2 = sigmaSq.x * cosL2 + sigmaSq.y * (1.0 - cosL2);

	float fresnel = 0.02 + 0.98 * pow(1.0 - dot(V, H), 5.0);

	zL = max(zL, 0.01);
	zV = max(zV, 0.01);

	return fresnel * p / ((1.0 + Lambda(zL, sigmaL2) + Lambda(zV, sigmaV2)) * zV * zH2 * zH2 * 4.0);
}

// ---------------------------------------------------------------------
// REFLECTED SKY RADIANCE
// ---------------------------------------------------------------------

// TODO?
// manual anisotropic filter

// V, N, Tx, Ty in world space
float2 U(float2 zeta, float3 V, float3 N, float3 Tx, float3 Ty)
{
	float3 f = normalize(float3(-zeta, 1.0)); // tangent space
	float3 F = f.x*Tx + f.y*Ty + f.z*N; // world space
	float3 R = 2.0 * dot(F, V) * F - V;
	return R.xz / (1.0 + R.y);
}

float meanFresnel(float cosThetaV, float sigmaV)
{
	return pow(1.0 - cosThetaV, 5.0 * exp(-2.69 * sigmaV)) / (1.0 + 22.7 * pow(sigmaV, 1.5));
}

// V, N in world space
float meanFresnel(float3 V, float3 N, float2 sigmaSq)
{
	float2 v = V.xz; // view direction in wind space
	float2 t = v * v / (1.0 - V.y * V.y); // cos^2 and sin^2 of view direction
	float sigmaV2 = dot(t, sigmaSq); // slope variance in view direction
	return meanFresnel(dot(V, N), sqrt(sigmaV2));
}

// V, N, Tx, Ty in world space
float3 meanSkyRadiance(float3 V, float3 N, float3 Tx, float3 Ty, float2 sigmaSq)
{
	float3 result = float3(0.0, 0.0, 0.0);

	const float eps = 0.001;
	float2 u0 = U(float2(0.0, 0.0), V, N, Tx, Ty);
	float2 dux = 2.0 * (U(float2(eps, 0.0), V, N, Tx, Ty) - u0) / eps * sqrt(sigmaSq.x);
	float2 duz = 2.0 * (U(float2(0.0, eps), V, N, Tx, Ty) - u0) / eps * sqrt(sigmaSq.y);

	result = skyMap.SampleGrad(samSkyMap, u0 * (0.5 / 1.1) + 0.5, dux * (0.5 / 1.1), duz * (0.5 / 1.1));
	// manual anisotropic
	// no filtering

	return result;
}


float4 main(VertexOut pin) : SV_TARGET
{
	//return float4(0.0, 0.0, 1.0, 1.0);
	float3 V = normalize(worldCamera - pin.PosW);

	float2 slopes = float2(0.0, 0.0);
	[unroll(4)]
	for (int i = 0; i < 4; ++i)
		slopes += gDisplacement.Sample(samFFTMap, float3(pin.u / GRID_SIZES[i], i/2 + 4)).xy;

	float3 N = normalize(float3(-slopes.x, 1.0, -slopes.y));
	if (dot(V, N) < 0.0)
		N = reflect(N, V);

	float Jxx = ddx(pin.u.x);
	float Jxy = ddy(pin.u.x);
	float Jyx = ddx(pin.u.y);
	float Jyy = ddy(pin.u.y);
	float A = Jxx * Jxx + Jyx * Jyx;
	float B = Jxx * Jxy + Jyx * Jyy;
	float C = Jxy * Jxy + Jyy * Jyy;
	const float SCALE = 10.0;
	float ua = pow(A / SCALE, 0.25);
	float ub = 0.5 + 0.5 * B / sqrt(A * C);
	float uc = pow(C / SCALE, 0.25);
	float2 sigmaSq = gSlopeVariance.Sample(samVariance, float3(ua, ub, uc)).rg;
	//return float4(sigmaSq, 0.0, 1.0);
	//return float4(Jxy*Jxy, 0, 0.0, 1.0)*1;
	// TODO: za duze Jxy. Why?!!
	sigmaSq = max(sigmaSq, 2e-5);

	float3 Ty = normalize(float3(0.0, N.z, -N.y));
	float3 Tx = cross(Ty, N);

	float fresnel = 0.02 + 0.98 * meanFresnel(V, N, sigmaSq);
	
	float3 Lsun;
	float3 Esky;
	float3 extinction;
	sunRadianceAndSkyIrradiance(worldCamera + earthPos, worldSunDir, Lsun, Esky);

	float3 result = float3(0.0, 0.0, 0.0);

	// SUN
	result += reflectedSunRadiance(worldSunDir, V, N, Tx, Ty, sigmaSq) * Lsun;
	
	// SKY
	result += fresnel * meanSkyRadiance(V, N, Tx, Ty, sigmaSq);

	// SEA
	float3 Lsea = seaColour * Esky / PI;
	result += (1.0 - fresnel) * Lsea;
	
	//result += 0.0001 * seaColour * (Lsun * max(dot(N, worldSunDir), 0.0) + Esky) / PI;

	return float4(HDR(result.rgb), 1.0);
}