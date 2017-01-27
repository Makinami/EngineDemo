#include "perFrameCB.hlsli" // b1

#include "..\WaterBruneton\atmosphere.hlsli"

cbuffer constBuffer : register(b0)
{
	float4 INVERSE_GRID_SIZE;
	float4 GRID_SIZE;
};

Texture2DArray wavesDisplacement : register(t0);
SamplerState samBilinear : register(s4);
SamplerState samAnisotropic : register(s1);
Texture2DArray turbulenceMap : register(t2);

sampler samTrilinear : register(s10);

Texture3D slopeVariance : register(t4);

Texture1DArray<float> fresnelTerm : register(t1);
SamplerState samClamp : register(s2);

Texture2D<float> foam : register(t13);
Texture2D<float> depth : register(t14);
Texture2D<float4> frame : register(t15);

Texture2D<float4> distanceField : register(t40);

Texture2D<float3> FoamCombined : register(t20);
Texture2D<float3> FoamRamp : register(t21);
Texture2D<float4> ColourDepth : register(t22);

//static const float4 seaColour = float4(10.0, 40.0, 120.0, 25.5) / 255.0;

//static const float3 rgbExtinction = float3(4.5, 75.0, 300.0) / 2.0;

static float Zview = 0.0f;
static float3 surfacePos = 0.0f.xxx;

struct DomainOut
{
	float4 PosH  : SV_POSITION;
	float4 PosF : TEXTCOORD0;
	float3 PosW : TEXTCOORD1;
	float4 params : TEXTCOORD2;
};

float meanFresnel(float cosThetaV, float sigmaV) {
	return pow(1.0 - cosThetaV, 5.0 * exp(-2.69 * sigmaV)) / (1.0 + 22.7 * pow(sigmaV, 1.5));
}

// V, N in world space
float meanFresnel(float3 V, float3 N, float2 sigmaSq) {
	float2 v = V.xz; // view direction in wind space
	float2 t = v * v / (1.0 - V.y * V.y); // cos^2 and sin^2 of view direction
	float sigmaV2 = dot(t, sigmaSq); // slope variance in view direction
	return meanFresnel(dot(V, N), sqrt(sigmaV2));
}

// Utility function that maps a value from one range to another.
float Remap(float original_value, float original_min, float original_max, float new_min, float new_max)
{
	return new_min + ((original_value - original_min) / (original_max - original_min)) * (new_max - new_min);
}

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

float3 hdr(float3 L) {
	L = L * 0.4;
	L.r = L.r < 1.413 ? pow(L.r * 0.38317, 1.0 / 2.2) : 1.0 - exp(-L.r);
	L.g = L.g < 1.413 ? pow(L.g * 0.38317, 1.0 / 2.2) : 1.0 - exp(-L.g);
	L.b = L.b < 1.413 ? pow(L.b * 0.38317, 1.0 / 2.2) : 1.0 - exp(-L.b);
	return L;
}

float4 main(DomainOut pin) : SV_TARGET
{
	float3 V = camPos - pin.PosW;
	float dist = length(V);
	V /= dist;

	float Zndc = depth[pin.PosH.xy];
	Zview = Zndc == 0.0f ? 1.0f / 0.0f : -(Zndc*gProj.w - gProj.z) / (Zndc*gProj.y - gProj.x);
	surfacePos = camPos + Zview * V;
	//if (pin.PosH.x > 640)
	//	return float4(max(Zview - pin.PosH.w, 0.0), 1.0.xxx);
	float waterDepthRay = Zview - pin.PosH.w;
	float3 ext = saturate(waterDepthRay / rgbExtinction);

	//return frame[pin.PosH.xy];

	float Amp = 0.27 * dot(gbWind, gbWind) / 9.81;
	Amp *= 0.5;

	float pi = 3.141529;
	float lambda = 10.0;// 2.0*pi;// A*2.0*pi; // minimal wavelength
	float k = 1.0 / (Amp * argHDependency); //2.0*pi / lambda;

	float sinp;
	float cosp;
	sincos(pin.params.x, sinp, cosp);

	float3 Normal;
	Normal.x = pin.params.z*pin.PosF.z*k*sinp;
	Normal.y = 1.0 - pin.PosF.z*k*(1.5*sinp + cosp)*pin.params.zw*pin.params.zw;
	Normal.z = pin.params.w*pin.PosF.z*k*sinp;
	Normal = normalize(Normal);

	float2 slopebig = 0.0.xx;
	float2 slopesmall = 0.0.xx;

	float3 resolution = float3(1.0 - smoothstep(2.0*GRID_SIZE.x, 4.0*GRID_SIZE.x, dist), 1.0 - smoothstep(1.0*GRID_SIZE.y, 3.0*GRID_SIZE.y, dist), 1.0 - smoothstep(0.5*GRID_SIZE.z, 4.0*GRID_SIZE.z, dist));
	slopebig += wavesDisplacement.Sample(samBilinear, float3(pin.PosF.xy / GRID_SIZE.x, 4)).xy;
	if (resolution.x > 0.0)
		slopebig += resolution.x*wavesDisplacement.Sample(samBilinear, float3(pin.PosF.xy / GRID_SIZE.y, 4)).zw;
	if (resolution.y > 0.0)
		slopesmall += resolution.y*wavesDisplacement.Sample(samBilinear, float3(pin.PosF.xy / GRID_SIZE.z, 5)).xy;
	if (resolution.z > 0.0)
		slopesmall += resolution.z*wavesDisplacement.Sample(samBilinear, float3(pin.PosF.xy / GRID_SIZE.w, 5)).zw;
	//return float4(saturate(resolution), 1.0);
	float2 slopeall = 0.0.xx;

	float3 NormalFFTBig = normalize(float3(-slopebig.x, 1.0, -slopebig.y));
	float3 NormalFFTSmall = normalize(float3(-slopesmall.x, 1.0, -slopesmall.y));

	//float3 NormalFFT = normalize(float3(-slopeall.x, 1.0, -slopeall.y));

	//Normal = normalize(lerp(NormalFFT, Normal, clamp(2.0*pin.params.y, 0.0, 0.8)));

	float3 NormalRough = normalize(lerp(NormalFFTBig, Normal, clamp(pin.params.y, 0.0, 0.8)));
	Normal = normalize(NormalRough + NormalFFTSmall);

	float3 N = Normal;

	float3 turbulence = float3(0.0, 0.0, 0.0);
	turbulence += turbulenceMap.Sample(samAnisotropic, float3(pin.PosF.xy / GRID_SIZE.x, 0)).xyz;
	if (resolution.x > 0.0)
		turbulence += resolution.x*turbulenceMap.Sample(samBilinear, float3(pin.PosF.xy / GRID_SIZE.y, 1)).xyz;
	if (resolution.y > 0.0)
		slopesmall += resolution.y*turbulenceMap.Sample(samBilinear, float3(pin.PosF.xy / GRID_SIZE.z, 2)).xyz;

	float3 R = reflect(V, N);
	float3 H = normalize(V + sunDir);

	float3 Tz = normalize(float3(0, N.z, -N.y));
	float3 Tx = cross(Tz, N);

	float Jxx = ddx(pin.PosF.x);
	float Jxy = ddy(pin.PosF.x);
	float Jyx = ddx(pin.PosF.y);
	float Jyy = ddy(pin.PosF.y);
	float A = Jxx * Jxx + Jyx * Jyx;
	float B = Jxx * Jxy + Jyx * Jyy;
	float C = Jxy * Jxy + Jyy * Jyy;
	float ua = pow(A / 10.0, 0.25);
	float ub = 0.5 + 0.5 * B / sqrt(A * C);
	float uc = pow(C / 10.0, 0.25);
	//float2 sigma = slopeVariance.Sample(samClamp, float2((ua + uc) / 2.0, (ua + uc) / 2.0)).rg; // Looks better but at what cost
	//float2 sigma = slopeVariance.Sample(samClamp, float2(ua, uc)).rg * 1.06;
	float2 sigma = slopeVariance.Sample(samClamp, float3(ua, ub, uc)).rg;
	sigma = max(sigma, 2e-5);
	//return float4(sigma*25, 0.0, 1.0);
	//return float4(sigma, 0.0, 1.0)*25.0;

	//return float4(sigma * 50, 0.0, 1.0);

	//float3 basicColour = float3(0.0, 0.0, dot(N, sunDir));
	//float3 sunLight = HDR(float3(1.0, 1.0, 1.0) * reflectedSunRadiance(sunDir, V, N, Tx, Ty, sigma2) * 90.0);

	//return float4(basicColour + sunLight, 1.0);
	//float fresnelUp = fresnelTerm.Sample(samClamp, dot(N, V), 0);
	float fresnelUp = saturate(0.02 + 0.98 * meanFresnel(V, N, sigma2));

	//float2 sigma = sigma2 * saturate(pow(Jxy*Jxy / 10.0 + Jyy*Jyy / 10.0, 0.25));

	float3 refractRay = refract(-V, NormalRough, 0.75);

	if (!any(refractRay))
	{
		refractRay = -V;
	}
	//return float4(refractRay.rb, fresnelUp, 1.0);
	float3 seenBottomPos = pin.PosW + refractRay*waterDepthRay;
	//return float4(seenBottomPos, 1.0);
	float4 seenBottomNDC = mul(float4(seenBottomPos, 1.0f), worldToScreenMatrix);
	float3 seenBottomScreenSpace = seenBottomNDC.xyz / seenBottomNDC.w;

	float3 seenBottom = frame.Sample(samTrilinear, (seenBottomScreenSpace.xy*float2(1.0,-1.0) + 1.0) / 2.0);

	float4 DF = distanceField.SampleLevel(samAnisotropic, (pin.PosW.xz + 2048.0f) / 4096.0, mipmap);

	float depthVertical = -DF.y;

	float depthFactor = pow(saturate(waterDepthRay / depthClamp), 0.5);

	float4 depthColour = ColourDepth.SampleLevel(samClamp, float2(depthFactor, 0.5), 0);

	float3 Lsun;
	float3 Esky;
	sunRadianceAndSkyIrradiance(camPos + earthPos, sunDir, Lsun, Esky);

	float3 result = float3(0.0, 0.0, 0.0);
	//return float4(0.0, 0.0, (-dot(sunDir, V))*dot(V,N)*(1.0 - NormalFFTBig.y)*100.0, 1.0);
	//float3 sunLight = fresnelUp * Lsun * smoothstep(cos(PI/10), cos(PI/30), dot(N, H)) * dot(N, H);
	float3 seaLight = 0.0.xxx;
	if (SeaFlag)
		seaLight = seaColour.rgb * seaColour.a * Esky / PI;
	//float3 seaLight = depthColour.rgb * colourAlpha * Esky / PI;
	float3 skyLight = 0.0.xxx;
	if (SkyFlag)
		skyLight = fresnelUp * meanSkyRadiance(V, N, Tx, Tz, sigma2);

	float3 sunLight = 0.0.xxx;
	if (SunFlag)
		sunLight = reflectedSunRadiance(sunDir, V, N, Tx, Tz, sigma) *Lsun;
	//return float4(seaLight, 1.0);

	// add sea light/bottom
	//if (SeaFlag)
	result += lerp(seaLight, seenBottom, 1 - ext) * (1.0 - fresnelUp);
	//result += lerp(seaLight, seenBottom, depthColour.a) * (1.0 - fresnelUp);
	//result += lerp(seaLight, frame[pin.PosH.xy], 1 - ext)*(1.0 - fresnelUp);
	//result += skyLight;
	//result += sunLight;
	//result += turbulence.zzz * (samNoise + turbulence.yyy);// *samNoise;

	float x = frac(pin.params.x / (2 * pi));
	float y1 = 3.0*x - 2;
	float y2 = -8 * x + 1;
	float foam_param = saturate(Remap(max(y1, y2), -pin.PosF.w/5.0, pin.PosF.w, 0.0, 1.0))*clamp(0.0, 1.5, pin.PosF.w*pin.PosF.w);
	//return float4(foam_param.xxx, 1.0);
	//saturate(max(y1, y2))*pin.PosF.z*pin.PosF.z;

	//return float4(foam_param.xxx, 1.0);
	// add sun light
	//if (SunFlag)
	//	result += sunLight;

	// add sky light
	//if (SkyFlag)
	//	result += skyLight;

	float3 globalLight = 0.0.xxx;
	if (any(int3(SeaFlag, SunFlag, SkyFlag)))
		globalLight += 0.0001 * seaColour * (Lsun * max(dot(N, sunDir), 0.0) + Esky) / PI;

	result += globalLight;

	float3 foam_texture = 0.0;
	float3 foam_ramp = 0.0;
	if (foam_param > 0.0)
	{
		foam_texture = FoamCombined.Sample(samBilinear, pin.PosF.xy / 10.0);
		foam_ramp = FoamRamp.Sample(samClamp, float2(foam_param, 0.5));
		//foam_texture = foam.Sample(samAnisotropic, pin.PosF.xy / 20.0);
		//foam_texture *= 0.5 + smoothstep(0.0, 2.0, 2.0 - turbulence.g);
		//foam_texture *= (foam_texture < (turbulence.z + foam_param) && (turbulence.z + foam_param) < 1.0) ? 0.0 : max(turbulence.z + foam_param, 1.0);// : 0.0;
		//foam_texture = foam_texture*(turbulance.z + foam_param)
		foam_texture.xyz = saturate(dot(foam_ramp, foam_texture));

		//result += lerp(skyLight + sunLight, foam_texture, foam_param);
	}
	else
	{
		//result += skyLight + sunLight;
	}

	//result += skyLight + sunLight;

	//result = result*(1.0 - foam_texture);
	//return float4(foam_texture, 1.0);
	//return 
	//result += (sunLight + skyLight) / (foam_texture.x*(PI - 1.0) + 1.0);

	//{
	//	result = foam_texture*Lsun*0.1 / PI;
	//	result += foam_texture*meanSkyRadiance(V, N, Tx, Tz, sigma2);
	//}

	result = 0.0.xxx;
	result += globalLight;
	result += lerp(seaLight, seenBottom, 1 - ext) * (foam_texture*0.05 / PI + (1.0 - foam_texture)*(1.0 - fresnelUp));
	result += meanSkyRadiance(V, N, Tx, Tz, sigma2)*(foam_texture + (1.0 - foam_texture)*fresnelUp); // SKY
	result += Lsun * (foam_texture + (1.0 - foam_texture)*reflectedSunRadiance(sunDir, V, N, Tx, Tz, sigma)); // SUN

																											  //result = globalLight;
																											  //result += skyLight + sunLight + sunLight*foam_texture;

																											  //if (foam_param > 0.0)
																											  //{
																											  //	foam_texture = foam.Sample(samAnisotropic, pin.PosF.xy / 15.0);
																											  //	//result = lerp(result, foam_text, foam_param);
																											  //}

																											  //if (turbulance.z > 0.0 || foam_param > 0.0)
																											  //{
																											  //	result += (skyLight + sunLight) * 0.1 + 
																											  //}

																											  //return float4(ua, ub, uc, 1.0);

	return float4(max(result, 0.0.xxx), 1.0);


	/*float3 view = normalize(camPos - pin.PosW);
	float diff = dot(sunDir, Normal);
	float specfactor = 0;
	if (diff > 0.0)
	{
	float3 v = reflect(-sunDir, Normal);
	specfactor = pow(max(dot(v, view), 0.0f), 4.0);
	}*/

	//return float4(specfactor.xxx, 1.0f);
}