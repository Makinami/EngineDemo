#include "perFrameCB.hlsli" // b1

#include "..\WaterBruneton\atmosphere.hlsli"

cbuffer constBuffer : register(b0)
{
	float4 INVERSE_GRID_SIZE;
	float4 GRID_SIZE;
};

Texture2DArray wavesDisplacement : register(t0);
SamplerState samAnisotropic : register(s1);

Texture1DArray<float> fresnelTerm : register(t1);
SamplerState samClamp : register(s2);

static const float4 seaColour = float4(10.0, 40.0, 120.0, 25.5) / 255.0;

struct VertexOut
{
	float4 PosH : SV_POSITION;
	float2 PosF : TEXTCOORD0;
	float3 PosW : TEXTCOORD1;
};

// assert x>0
float erfc(float x)
{
	return 2.0 * exp(-x * x) / (2.319 * x + sqrt(4.0 + 1.52 * x * x));
}

float Lambda(float a)
{
	return (exp(-a*a) - a*sqrt(PI)*erfc(a)) / (2.0 * a * sqrt(PI));
}

float Lambda(float cosTheta, float sigmaSq) {
	float v = cosTheta / sqrt((1.0 - cosTheta * cosTheta) * (2.0 * sigmaSq));
	return max(0.0, (exp(-v * v) - v * sqrt(PI) * erfc(v)) / (2.0 * v * sqrt(PI)));
	//return (exp(-v * v)) / (2.0 * v * sqrt(M_PI)); // approximate, faster formula
}

float reflectedSunRadiance(float3 L, float3 V, float3 N, float3 Tx, float3 Tz, float2 sigma2)
{
	/*float3 H = normalize(V + L);
	float zetax = dot(H, Tx) / dot(H, N);
	float zetay = dot(H, Tz) / dot(H, N);
	
	float p = exp(-0.5 * (zetax * zetax / sigma2.x + zetay * zetay / sigma2.y)) / (2.0 * PI * sqrt(sigma2.x * sigma2.y));

	float tanV = atan2(dot(V, Tz), dot(V, Tx));
	float cosV2 = 1.0 / (1.0 + tanV * tanV);
	float aV = 1.0 / sqrt(2.0 * (sigma2.x * cosV2 + sigma2.y * (1.0 - cosV2)) * tanV);

	float tanL = atan2(dot(L, Tz), dot(L, Tx));
	float cosL2 = 1.0 / (1.0 + tanL * tanL);
	float aL = 1.0 / sqrt(2.0 * (sigma2.x * cosL2 + sigma2.y * (1.0 - cosL2)) * tanL);

	float fresnel = 0.02 + 0.98 * pow(1.0 - dot(V, H), 5.0);

	float Hz = dot(H, N);
	float Hz2 = Hz * Hz;
	float Vz = dot(V, N);

	return fresnel * p / ((1.0 + Lambda(aV) + Lambda(aL)) * 4.0 * Hz2 * Hz2 * Vz);*/

	float3 H = normalize(L + V);
	float zetax = dot(H, Tx) / dot(H, N);
	float zetay = dot(H, Tz) / dot(H, N);

	float zL = dot(L, N); // cos of source zenith angle
	float zV = dot(V, N); // cos of receiver zenith angle
	float zH = dot(H, N); // cos of facet normal zenith angle
	float zH2 = zH * zH;

	float p = exp(-0.5 * (zetax * zetax / sigma2.x + zetay * zetay / sigma2.y)) / (2.0 * PI * sqrt(sigma2.x * sigma2.y));

	float tanV = atan2(dot(V, Tz), dot(V, Tx));
	float cosV2 = 1.0 / (1.0 + tanV * tanV);
	float sigmaV2 = sigma2.x * cosV2 + sigma2.y * (1.0 - cosV2);

	float tanL = atan2(dot(L, Tz), dot(L, Tx));
	float cosL2 = 1.0 / (1.0 + tanL * tanL);
	float sigmaL2 = sigma2.x * cosL2 + sigma2.y * (1.0 - cosL2);

	float fresnel = 0.02 + 0.98 * pow(1.0 - dot(V, H), 5.0);

	zL = max(zL, 0.01);
	zV = max(zV, 0.01);

	return fresnel * p / ((1.0 + Lambda(zL, sigmaL2) + Lambda(zV, sigmaV2)) * zV * zH2 * zH2 * 4.0);
}

float2 U(float2 zeta, float3 V, float3 N, float3 Tx, float3 Ty)
{
	float3 f = normalize(float3(-zeta.x, 1.0, -zeta.y));
	float3 F = f.x * Tx + f.y * N + f.z * Ty;
	float3 R = 2.0 * dot(F, V) * F - V;
	return R.xz / (1.0 + R.y);
}

float3 meanSkyRadiance(float3 V, float3 N, float3 Tx, float3 Ty, float2 sigma2)
{
	const float eps = 0.001;

	float2 u0 = U(float2(0.0, 0.0), V, N, Tx, Ty);
	float2 dux = 2.0 * (U(float2(eps, 0.0), V, N, Tx, Ty) - u0) / eps * sqrt(sigma2.x);
	float2 duy = 2.0 * (U(float2(0.0, eps), V, N, Tx, Ty) - u0) / eps * sqrt(sigma2.y);

	return skyMap.SampleGrad(samAnisotropic, u0 * (0.5 / 1.1) + 0.5, dux * (0.5 / 1.1), duy * (0.5 / 1.1));
}

float4 main( VertexOut pin ) : SV_TARGET
{
	float3 V = normalize(camPos - pin.PosW);

	float2 slope = float2(0.0, 0.0);
	slope += wavesDisplacement.Sample(samAnisotropic, float3(pin.PosF / GRID_SIZE.x, 4)).xy;
	slope += wavesDisplacement.Sample(samAnisotropic, float3(pin.PosF / GRID_SIZE.y, 4)).zw;
	slope += wavesDisplacement.Sample(samAnisotropic, float3(pin.PosF / GRID_SIZE.z, 5)).xy;
	slope += wavesDisplacement.Sample(samAnisotropic, float3(pin.PosF / GRID_SIZE.w, 5)).zw;
	
	float3 N = normalize(float3(-slope.x, 1.0, -slope.y));
	float3 R = reflect(V, N);
	float3 H = normalize(V + sunDir);

	float3 Tz = normalize(float3(0, N.z, -N.y));
	float3 Tx = cross(Tz, N);

	//float3 basicColour = float3(0.0, 0.0, dot(N, sunDir));
	//float3 sunLight = HDR(float3(1.0, 1.0, 1.0) * reflectedSunRadiance(sunDir, V, N, Tx, Ty, sigma2) * 90.0);

	//return float4(basicColour + sunLight, 1.0);

	float fresnelUp = fresnelTerm.Sample(samClamp, dot(N, V), 0);

	float3 Lsun;
	float3 Esky;
	sunRadianceAndSkyIrradiance(camPos + earthPos, sunDir, Lsun, Esky);

	float3 result = float3(0.0, 0.0, 0.0);

	float3 sunLight = fresnelUp * Lsun * step(cos(PI/15), dot(N, H)) * dot(N, H);
	float3 seaLight = (1.0 - fresnelUp) * seaColour.rgb * seaColour.a * Esky / PI;
	float3 skyLight = fresnelUp * meanSkyRadiance(V, N, Tx, Tz, sigma2);
	//float3 sunLight = reflectedSunRadiance(sunDir, V, N, Tx, Tz, sigma2) * fresnelUp * Lsun;

	//result += seaLight;
	//result += skyLight;
	result += sunLight;

	return float4(HDR(result), 1.0);
}