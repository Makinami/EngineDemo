Texture2D<float3> transmittance : register(t1);
TextureCube gCubeMap : register(t3);

Texture3D<float4> CloudsLFNoises;

Texture3D<float3> CloudsHFNoises;

//Texture2D<float4> cloudsCurl : register(t6);
//Texture2D<float4> cloudsType : register(t7);
Texture2D<float4> weather : register(t8);

float GetNormalizedHeightInClouds(in float3 pos, in float2 cloudRange)
{
	float height = (pos.z - cloudRange.x) / (cloudRange.y - cloudRange.x);
	return saturate(height);
}

float Remap(float orig_val, float orig_min, float orig_max, float new_min, float new_max)
{
	return new_min + (orig_val - orig_min) / (orig_max - orig_min) * (new_max - new_min);
}

float SampleCloudDensity(float3 p, float3 weather, float mipLevel)
{
	// wind settings
	float3 windDir = float3(1.0, 0.0, 0.0);
	float cloudSpeed = 10.0;

	// cloud top offset pushes the top this many unit
	float cloudTopOffset = 500.0;

	// get height fraction
	float heightNormal = GetNormzalizedHeightInClouds(p, cloudRange);

	// skew in wind direction
	p += heighNormal * windDir * cloudTopOffset;

	// animate in wind direction
	p += (windDir + float3(0.0, 0.1, 0.0)) * time * cloudSpeed;

	// get low frequency noises
	float4 LFnoises = CloudsLFNoises.SampleLod(CloudsNoisesSampler, p, mipLevel).rgba;

	// build low frequency FBM
	float LF_FBM = (LFnoises.g * 0.625) + (LFnoises.b * 0.25) + (LFnoises.a * 0.125);

	// base cloud shape
	float baseCloud = Remap(LFnoises.r, 1.0 - LF_FBM, 1.0, 0.0, 1.0);

	// density-heigh gradient
	float densityHeighGradient = GetDensityHeighGradient(p, weather);

	// apply gradient
	baseCloud *= densityHeightGradient;

	// cloud coverage - weather.r
	float cloudCoverage = weather.r;

	// remap to include cloud coverage
	float baseCloudWithCoverage = Remap(baseCloud, cloudCoverage, 1.0, 0.0, 1.0);

	// 
	baseCloudWithCoverage *= cloudCoverage;
	
	// add turbulence to the bottoms of clouds
	p.xy += curlNoise.xy * (1.0 - heightNormal);

	float3 HFnoises = CloudsHFNoises.SampleLod(CloudsNoisesSampler, p*0.1, mipLevel).rgb;

	// built heigh frequency noise
	float HF_FBM = (HFnoises.r * 0.625) + (HFnoises.g * 0.25) + (HFnoises.b * 0.125);

	// wispy-billowy transition
	float HFnoiseModifier = mix(HF_FBM, 1.0 - HF_FBM, saturate(heightNormal * 10.0));

	// erode base cloud shape with high-frequency Worley noise
	float finalCloud = Remap(baseCloudWithCoverage, HFnoiseModifier * 0.2, 1.0, 0.0, 1.0);

	return finalCloud;
}

float Energy(float d, float p, float g, float ctheta)
{
	return 2.0 * exp(-d*p) * (1.0 - exp(-2 * d)) * (1.0 - g*g) / pow(1.0 + g*g - 2 * g*ctheta, 3.0 / 2.0) / (4.0 * XM_PI);
}

float4 main() : SV_TARGET
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



	return float4(1.0f, 1.0f, 1.0f, 1.0f);
}