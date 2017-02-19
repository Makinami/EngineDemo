Texture3D<float4> inscatter : register(t0);
Texture2D<float3> transmittance : register(t1);
Texture2D<float4> deltaE : register(t2);
TextureCube gCubeMap : register(t3);
Texture3D<float4> cloudsGeneral : register(t4);
Texture3D<float4> cloudsDetail : register(t5);
Texture2D<float4> cloudsCurl : register(t6);
Texture2D<float4> cloudsType : register(t7);
Texture2D<float4> weatherPar : register(t8);

static const float Rl = 6361.5;
static const float Rh = 6365.0;

SamplerState samInscatter : register(s0);
SamplerState samTransmittance : register(s1);

SamplerState samTrilinearSam : register(s3);
//SamplerState samBilinearClamp : register(s4);

#define FIX

#define USE_TRANSMITTANCE
#define USE_INSCATTER

#pragma warning(disable:3568)
//#include <..\Sky\resolutions.h>
#pragma warning(default:3568)

#include <..\Sky2\Common.hlsli>

float3 getTransmittanceWithShadow(float r, float mu)
{
	return mu < -sqrt(1.0 - (groundR / r)*(groundR / r)) ? float3(0.0, 0.0, 0.0) : getTransmittance(r, mu);
}

static const float ISun = 100.0;
static const float G_SCATTERING = 0.9f;

static float3 sunLight;
static float3 skyLight;
static float3 bSunDir1;
static float cosViewSun;
static float4 weatherData;

static float3 viewRay;

static const float3 absorptionFactor = float3(2., 1.5, 1.0);
static const float absFac = 2.0;
static const float scatteringCoef = 0.25;
static const float3 scatteringFactor = float3(1.0, 1.0, 1.0);

/*
dystrybuanta gaussa
x = [ 0, 2.2] 0.1*x*(4.4 - x) + 0.5
x = [-2.2, 0] 0.1*x*(4.4 + x) + 0.5
odwrotnosc dystrybuanty gaussu
y = [0.5, 0.984]  2.2 - sqrt(4.84 - 10*(y - 0.5))
y = [0.016, 0.5] -2.2 + sqrt(4.84 + 10*(y - 0.5))
*/

static const float3 noise_kernel[] =
{
	float3(-0.393554, 0.0388882, 0.918478),
	float3(-0.497004, -0.80077, -0.334296),
	float3(0.676772, 0.735482, 0.0323357),
	float3(0.860109, -0.484374, -0.15998),
	float3(-0.301109, -0.924295, -0.234545),
	float3(0.789508, -0.480101, 0.382335),
	float3(0.756813, -0.585189, -0.291186)
};

cbuffer cbPerFramePS
{
	float3 bCameraPos;
	float bExposure;
	float3 bSunDir;
	float pad;
	float4 parameters[10];
};

#define F4_COVERAGE(f) f.r
#define F4_RAIN(f) f.g
#define F4_TYPE(f) f.b

#define WEATHER_PARAMS saturate(parameters[0])

#define WEATHER_COVERAGE F4_COVERAGE(WEATHER_PARAMS)
#define WEATHER_RAIN F4_RAIN(WEATHER_PARAMS)
#define WEATHER_TYPE F4_TYPE(WEATHER_PARAMS)

#define CUMULONIMBUS_BAND parameters[1]
#define CUMULUS_BAND parameters[2]
#define STRATUS_BAND parameters[3]

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
		float3 transmittance = r <= topR ? getTransmittanceWithShadow(r, mu) : float3(1.0, 1.0, 1.0); // T(x,x0)
		float isun = step(cos(PI / 180.0), dot(v, s)) * ISun; // Lsun dot(s, v);// 
		return transmittance*isun; // Eq (9)
	}
}

float3 HDR(float3 L)
{
	L = L*0.4;
	L.r = L.r < 1.413 ? pow(L.r * 0.38317, 1.0 / 2.2) : 1.0 - exp(-L.r);
	L.g = L.g < 1.413 ? pow(L.g * 0.38317, 1.0 / 2.2) : 1.0 - exp(-L.g);
	L.b = L.b < 1.413 ? pow(L.b * 0.38317, 1.0 / 2.2) : 1.0 - exp(-L.b);
	return L;
}

float BandPass(float x, float4 filter)
{
	return smoothstep(filter.x, filter.y, x) - smoothstep(filter.z, filter.w, x);
}

float Lerp3(float v0, float v1, float v2, float x)
{
	return x < 0.5 ? lerp(v0, v1, 2.0*x) : lerp(v1, v2, (x - 0.5)*2.0);
}

float4 Lerp3Smoothstep(float4 v0, float4 v1, float4 v2, float x)
{
	return x < 0.5 ? lerp(v0, v1, smoothstep(0.0, 1.0, 2.0*x)) : lerp(v1, v2, smoothstep(0.0, 1.0, (x - 0.5)*2.0));
}

// remaping and calmping value to new range
float Remap(float original_value, float original_min, float original_max, float new_min, float new_max)
{
	return new_min + (original_value - original_min) / (original_max - original_min) * (new_max - new_min);
}

// height in cloud [0..1]
float GetHeightFractionForPoint(in float3 pos)
{
	float height_fraction = (length(pos) - Rl) / (Rh - Rl);

	return saturate(height_fraction);
}

float GetDensityHeightGradient(float3 p)
{
	float cloudType = WEATHER_TYPE;
	float normalized_height = GetHeightFractionForPoint(p);

	float4 cloudsBand = Lerp3Smoothstep(STRATUS_BAND, CUMULUS_BAND, CUMULONIMBUS_BAND, cloudType);

	return BandPass(normalized_height, cloudsBand);
}

// get cloud density from pos
float SampleCloudDensity(in float3 pos, in bool detailed = false)
{
	// read low-frequency noise 
	float base_cloud = cloudsGeneral.SampleLevel(samTrilinearSam, pos / 5, 0);

	float density_height_gradient = GetDensityHeightGradient(pos);

	base_cloud *= density_height_gradient;
	//return base_cloud;
	//return smoothstep(parameters.x, parameters.y, base_cloud);
	weatherData = weatherPar.SampleLevel(samTrilinearSam, (pos.xz - 35.0) / 70.0, 0);
	float cloud_coverage = WEATHER_COVERAGE;// weatherData.r;

	float base_cloud_with_coverage = pow(saturate(Remap(base_cloud, 1.0 - cloud_coverage, 1.0, 0.0, 1.0)), 0.125);

	//float base_cloud_with_coverage = (base_cloud < 1.0 - cloud_coverage ? 0.0 : base_cloud);
	base_cloud_with_coverage *= cloud_coverage;
	//float base_cloud_with_coverage = smoothstep(1.0 - cloud_coverage, 1.0, base_cloud);

	if (!detailed) return (base_cloud_with_coverage);

	float high_frequency_noises = cloudsDetail.SampleLevel(samTrilinearSam, pos / (16 * 5.0), 0);
	float high_freq_FBM = high_frequency_noises;
	//float high_freq_FBM = dot(high_frequency_noises, float3(0.625, 0.25, 0.125));

	float heigh_fraction = GetHeightFractionForPoint(pos);

	float high_freq_noise_modifier = lerp(high_freq_FBM, 1.0 - high_freq_FBM, saturate(heigh_fraction * 10.0));

	float final_cloud = Remap(base_cloud_with_coverage, high_freq_noise_modifier * 0.2, 1.0, 0.0, 1.0);

	return (final_cloud);
}

// direct light reaching point pos from global bSunDir1
float3 GetDirectSunlight(float3 pos)
{
	float r = length(pos);
	float mu = dot(pos, bSunDir1) / r;
	float tg = -r * mu + sqrt(r * r * (mu * mu - 1.0f) + groundR * groundR);

	if (tg > 0.0)
		return float3(0.0, 0.0, 0.0);
	else
	{
		float3 transmittance = r <= topR ? getTransmittanceWithShadow(r, mu) : float3(1.0, 1.0, 1.0);
		return transmittance*ISun;
	}
}

// Henyey-Greenstein phase function
float HenyeyGreenstein(float cos_theta, float inG)
{
	return ((1.0 - inG * inG) / pow((1.0 + inG * inG - 2.0 * inG * cos_theta), 3.0 / 2.0)) / (4.0 * 3.14159);
}

float TwoLobePhase(float cos_theta, float g0 = 0.8, float g1 = -0.5, float alpha = 0.5)
{
	return lerp(HenyeyGreenstein(cos_theta, g0), HenyeyGreenstein(cos_theta, g1), alpha);
}

// Exponential Integral
// (http://en.wikipedia.org/wiki/Exponential_integral)
float Ei(float z)
{
	return 0.5772156649015328606065 + log(1e-4 + abs(z)) + z * (1.0 + z * (0.25 + z * ((1.0 / 18.0) + z * ((1.0 / 96.0) + z * (1.0 / 600.0))))); // For z != 0
}

float3 IncidentLighting(in float3 pos, bool detailed)
{
	float density = 0.0;
	//float3 sunLight = GetDirectSunlight(pos);

	float3 lightStep = -bSunDir1 * 0.1; // NOTE: different light step?
	float coneSpreadMultiplier = length(lightStep);

	// short cone
	[unroll(6)]
	for (uint i = 0; i <= 5; ++i)
	{
		pos += lightStep + coneSpreadMultiplier * noise_kernel[i] * float(i);
		density += SampleCloudDensity(pos, detailed) * 0.2;
	}

	// one long sample
	//density += SampleCloudDensity(pos + lightStep * 1.8, detailed) * 1.8; // max(cloudsGeneral.SampleLevel(samTrilinearSam, pos + lightStep * 1.8, 0), 0);

	//float HG = HenyeyGreenstein(cosViewSun, 0.2);
	float chances_of_rain = 0.25;
	float rain = chances_of_rain * chances_of_rain * 10 + 1.0;

	return sunLight * exp(-rain * density) * (1.0 - exp(-density * 2.0)) * 2.0 *(SampleCloudDensity(pos + lightStep * 1.8, detailed) > 0.5 ? 0.5 : 1.0);
	//return 2.0 * sunLight * HG * exp(-absFac * density) * (1.0 - exp(-absFac * density * 2));
}

float CloudSelfShadow(in float3 pos, bool detailed)
{
	float extinction = 0.0;

	float lightStep = bSunDir1 * 0.05;
	float coneSpreadAngle = PI / 8;
	float coneSpreadAngleTg = tan(coneSpreadAngle);

	// short cone
	[unroll(6)]
	for (uint i = 0; i <= 5; ++i)
	{
		extinction += SampleCloudDensity(pos + lightStep * (i + 1) * coneSpreadAngleTg * noise_kernel[i], detailed) * lightStep;
	}

	extinction += SampleCloudDensity(pos + lightStep * 0.9, detailed) * 0.6;

	return exp(-extinction)*(1.0 - exp(-2.0*extinction));
}

float3 AmbientLighting(in float3 pos, float density)
{
	float avg_density = density * weatherData.r * 0.95 * weatherData.g / 2.0;
	float alpha = avg_density * (Rh - length(pos));
	float3 IsotropicScatteringTop = skyLight * max(0.0, exp(alpha) - alpha * Ei(alpha));

	return IsotropicScatteringTop * 0;
}

float4 MarchClouds(in float3 camera, in float3 ray, in float from, in float to)
{
	float3 pos = camera + ray * from;
	float r = length(pos);
	// cos zenith angle
	float cosViewZenith = dot(pos, viewRay) / r;

	uint numsteps = ceil(64 * (2.0 - abs(cosViewZenith)));
	float ds = (to - from) / float(numsteps);
	float3 step = ray * ds;

	float transmittance = 1.0;
	float ext;
	float3 scatteredLight = float3(0.0, 0.0, 0.0);

	sunLight = GetDirectSunlight(camera + ray * from);

	float sunPhase = TwoLobePhase(cosViewSun);
	float skyPhase = TwoLobePhase(cosViewZenith);

	for (float t = from; t <= to; )
	{
		pos = camera + ray * t;

		ext = SampleCloudDensity(pos, true);

		if (ext > 0.0)
		{
			float3 S = sunLight*sunPhase*CloudSelfShadow(pos, false);
			float3 Sint = (S - S*exp(-ext*ds)) / ext;
			scatteredLight += Sint * transmittance;

			transmittance *= exp(-ext * ds);

			if (transmittance < 1e-2)
				break;
		}

		t += ds;
	}

	return float4(scatteredLight, 1.0 - transmittance);
}

float3 GetInscatter(in float3 viewDir, out float3 attenuation, inout float irradianceFactor, float pathLength, float3 surfacePos)
{
	float3 inscatteredLight = 0.0f.xxx;

	attenuation = 0.0f.xxxx;
	float offset;

	if (true)
	{
		// offset camera
		float3 startPos = bCameraPos;
		float startPosR = length(startPos);
		// now startPos definitely inside atmosphere

		float vzAngleStart = dot(startPos, viewDir) / startPosR;
		float vsAngle = dot(viewDir, bSunDir);
		float szAngleStart = dot(startPos, bSunDir) / startPosR;

		float4 inscatter = getInscatter(startPosR, vzAngleStart, szAngleStart, vsAngle);
		return inscatter.rgb * 100;
		//return inscatter;
		float surfacePosR = length(surfacePos);
		float szAngleEnd = dot(surfacePos, bSunDir) / surfacePosR;
		//return (maxPathLength - pathLength).xxx;
		// if surface if inside the atmosphere
		if (true)
		{
			// reduce inscatter light to start-surface path
			attenuation = analyticTransmittance(startPosR, vzAngleStart, pathLength);

			float vzAngleEnd = dot(surfacePos, viewDir) / surfacePosR;
			float4 inscatterAtSurface = getInscatter(surfacePosR, vzAngleEnd, szAngleEnd, vsAngle);

			inscatter = max(inscatter - attenuation.rgbr*inscatterAtSurface, 0.0f);
			irradianceFactor = 1.0f;
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

float4 main(PixelInputType pin) : SV_TARGET
{
	viewRay = normalize(pin.Ray);
float3 posRelSun = bCameraPos / 1000.0 + float3(0.0, groundR, 0.0); // TODO: handling arbitrary position of Earth

																	// length from core
float r = length(posRelSun);
// cos zenith angle
float mu = dot(posRelSun, viewRay) / r;
float smu = sign(mu);

// TODO: better distance calculation; these are all over the place when grazing angle

// distance to ground along viewing ray
float tGround = -r * mu + smu * sqrt(r * r * (mu * mu - 1.0f) + groundR * groundR);

// distance to base/top of clouds
float tCloudBase = -r * mu + smu * sqrt(r * r * (mu * mu - 1.0f) + Rl * Rl);
float tCloudTop = -r * mu + smu * sqrt(r * r * (mu * mu - 1.0f) + Rh * Rh);

if (tCloudBase > tCloudTop)
{
	// TODO: check if packing into float2 is faster
	float temp = tCloudBase;
	tCloudBase = tCloudTop;
	tCloudTop = temp;
}

// clip if above clouds, and looking up
//clip(tCloudTop);
if (tCloudTop < 0.0) return float4(0.0, 0.0, 0.0, 0.0);

// don't add what's behind the camera
tCloudBase = max(tCloudBase, 0.0);

if (tGround > 0.0)
tCloudTop = min(tCloudTop, tGround);

//clip(tCloudTop > tCloudBase ? 1.0 : -1.0);
if (tCloudTop < tCloudBase) return float4(0.0, 0.0, 0.0, 0.0);

// set global sun direction vector
bSunDir1 = -normalize(bSunDir);
cosViewSun = dot(viewRay, bSunDir1);

float4 clouds = MarchClouds(posRelSun, viewRay, tCloudBase, tCloudTop);

float3 attenuation = 0.0f.xxx;
float irradianceFactor = 0.0f;

float3 inscatterLight = GetInscatter(viewRay, attenuation, irradianceFactor, tCloudBase, bCameraPos + viewRay*tCloudBase);

clouds.rgb = inscatterLight;
clouds.rgb = HDR(clouds.rgb);
clouds.a = 1.0;

return clouds;
}