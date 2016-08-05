Texture3D<float4> inscatterTex : register(t0);
Texture2D<float3> transmittance : register(t1);
TextureCube gCubeMap : register(t3);
Texture3D<float> cloudsGeneral : register(t4);
Texture3D<float> cloudsDetail : register(t5);
Texture2D<float4> cloudsCurl : register(t6);
Texture2D<float4> cloudsType : register(t7);
Texture2D<float4> weatherPar : register(t8);

static const float Rl = 6361.5;
static const float Rh = 6364.0;

SamplerState samInscatter : register(s0);
SamplerState samTransmittance : register(s1);

SamplerState samTrilinearSam : register(s3);

#define FIX

#define USE_TRANSMITTANCE
#define USE_INSCATTER

#pragma warning(disable:3568)
#include <..//Sky//resolutions.h>
#pragma warning(default:3568)

#include <..\Sky\common.hlsli>

static const float ISun = 100.0;
static const float G_SCATTERING = 0.9f;

static float3 sunLight;
static float3 skyLight;
static float3 bSunDir1;
static float cosViewSun;
static float4 weatherData;

static float3 viewRay;

static const float3 absorptionFactor = float3(1.5, 2.0, 2.5);
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

float3 HDR(float3 L)
{
	L = L*0.4;
	L.r = L.r < 1.413 ? pow(L.r * 0.38317, 1.0 / 2.2) : 1.0 - exp(-L.r);
	L.g = L.g < 1.413 ? pow(L.g * 0.38317, 1.0 / 2.2) : 1.0 - exp(-L.g);
	L.b = L.b < 1.413 ? pow(L.b * 0.38317, 1.0 / 2.2) : 1.0 - exp(-L.b);
	return L;
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
	float normalized_height = GetHeightFractionForPoint(p);
	float density = cloudsType.SampleLevel(samTrilinearSam, float2(weatherData.g, normalized_height), 0).r;
	// TODO: more dependent on the weather
	return density * (1.0 - normalized_height);
}

// get cloud density from pos
float SampleCloudDensity(in float3 pos, in bool detailed = false)
{
	// read low-frequency noise 
	float base_cloud = cloudsGeneral.SampleLevel(samTrilinearSam, pos / 5, 0);

	float density_height_gradient = GetDensityHeightGradient(pos);

	base_cloud *= density_height_gradient;

	weatherData = weatherPar.SampleLevel(samTrilinearSam, (pos.xz - 35.0) / 70.0, 0);
	float cloud_coverage = weatherData.b;

	float base_cloud_with_coverage = Remap(base_cloud, 1.0 - cloud_coverage, 1.0, 0.0, 1.0);
	//float base_cloud_with_coverage = (base_cloud < 1.0 - cloud_coverage ? 0.0 : base_cloud);
	base_cloud_with_coverage *= cloud_coverage;
	//float base_cloud_with_coverage = smoothstep(1.0 - cloud_coverage, 1.0, base_cloud);

	if (!detailed) return saturate(base_cloud_with_coverage);

	float high_frequency_noises = cloudsDetail.SampleLevel(samTrilinearSam, pos / (16 * 5.0), 0);
	float high_freq_FBM = high_frequency_noises;
	//float high_freq_FBM = dot(high_frequency_noises, float3(0.625, 0.25, 0.125));

	float heigh_fraction = GetHeightFractionForPoint(pos);

	float high_freq_noise_modifier = lerp(high_freq_FBM, 1.0 - high_freq_FBM, saturate(heigh_fraction * 10.0));

	float final_cloud = Remap(base_cloud_with_coverage, high_freq_noise_modifier * 0.2, 1.0, 0.0, 1.0);

	return saturate(final_cloud);
}

// direct light reaching point pos from global bSunDir1
float3 GetDirectSunlight(float3 pos)
{
	float r = length(pos);
	float mu = dot(pos, bSunDir1) / r;
	float tg = -r * mu + sqrt(r * r * (mu * mu - 1.0f) + Rg * Rg);

	if (tg > 0.0)
		return 0.0.xxx;
	else
	{
		float3 transmittance = r <= Rt ? getTransmittanceWithShadow(r, mu) : float3(1.0, 1.0, 1.0);
		return transmittance*ISun;
	}
}

// Henyey-Greenstein phase function
float HenyeyGreenstein(float cos_theta, float inG)
{
	return ((1.0 - inG * inG) / pow((1.0 + inG * inG - 2.0 * inG * cos_theta), 3.0 / 2.0)) / (4.0 * 3.14159);
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

	float3 lightStep = bSunDir1 * 0.1; // NOTE: different light step?
	float coneSpreadMultiplier = length(lightStep);

	// short cone
	[unroll(6)]
	for (uint i = 0; i <= 5; ++i)
	{
		pos += lightStep + coneSpreadMultiplier * noise_kernel[i] * float(i);
		density += SampleCloudDensity(pos, detailed) * 0.3;
	}

	// one long sample
	density += SampleCloudDensity(pos + lightStep * 1.8, detailed) * 1.8; // max(cloudsGeneral.SampleLevel(samTrilinearSam, pos + lightStep * 1.8, 0), 0);

																		  //float HG = HenyeyGreenstein(cosViewSun, 0.2);
	float rain = 1.0;

	return sunLight * exp(-rain * density) * (1.0 - exp(-density * 2.0)) * 2.0;
	//return 2.0 * sunLight * HG * exp(-absFac * density) * (1.0 - exp(-absFac * density * 2));
}

float3 AmbientLighting(in float3 pos, float density)
{
	float avg_density = density * weatherData.g / 2.0;
	float alpha = avg_density * (Rh - length(pos));
	float3 IsotropicScatteringTop = skyLight * max(0.0, exp(alpha) - alpha * Ei(alpha));

	return IsotropicScatteringTop;
}

// inscattered ligth taking ray x+tv, when the Sun in direction s (=S[L]-T(x, x0)S[L]x0)
float3 inscatter(in float3 x, in float t, float3 v, float3 s, out float r, out float mu, out float3 attenuation)
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


float4 MarchClouds(in float3 camera, in float3 ray, in float from, in float to)
{
	float3 pos = camera + ray * from;
	float r = length(pos);
	// cos zenith angle
	float mu = dot(pos, viewRay) / r;

	uint numsteps = ceil(64 * (2.0 - abs(mu)));
	float ds = (to - from) / float(numsteps);
	//ds = 0.05; numsteps = ceil((to - from) / ds);
	float3 step = ray * ds;

	float3 transmittanceAll = 1.0.xxx;
	float3 transmittance;
	float opacity = 0.0;
	float3 colour = 0.0.xxx;

	float3 incidentLight;
	float3 ambientLight;

	sunLight = GetDirectSunlight(pos);
	float3 attenuation = 0.0.xxx;
	skyLight = inscatter(pos, Rg - r, normalize(pos), bSunDir1, r, mu, attenuation);
	weatherData = weatherPar.SampleLevel(samTrilinearSam, (pos.xz - 35.0) / 70.0, 0);

	float density = 0.0;
	bool detailed = 0.0;
	uint zero_density_sample_count = 0;

	for (float t = from; t <= to; )
	{
		pos = camera + ray * t;

		density = SampleCloudDensity(pos, detailed);

		if (density == 0.0)
			++zero_density_sample_count;

		if (detailed)
		{
			if (zero_density_sample_count >= 6)
			{
				zero_density_sample_count = 0;
				detailed = false;
				ds *= 2.5;
				t += ds;
				continue;
			}
		}
		else
		{
			if (density > 0.0 && opacity < 0.3)
			{
				ds /= 2.5;
				detailed = true;
				zero_density_sample_count = 0;
				continue;
			}
		}

		if (density > 0.0)
		{
			// Lighting
			transmittance = exp(-absorptionFactor * ds * density);
			transmittanceAll *= transmittance;

			incidentLight = IncidentLighting(pos, detailed);
			ambientLight = AmbientLighting(pos, density);
			colour += (1.0 - transmittance) * scatteringFactor * (incidentLight * HenyeyGreenstein(cosViewSun, 0.2) + ambientLight * HenyeyGreenstein(mu, 0.2)) * transmittanceAll;
			// TODO: wyrzuci? oba HG poza loop (const per pixel)
			opacity += (1.0 - dot(transmittance, 1.0.xxx) / 3.0) * (1.0 - opacity);
		}

		// if opaque no point in further raymarching
		if (opacity > 0.99)
		{
			opacity = 1.0;
			break;
		}

		// move
		t += ds;

		//density = 8 * SampleCloudDensity(pos, false);// max(cloudsGeneral.SampleLevel(samTrilinearSam, pos, 0), 0);
		//if (density == 0.0)
		//{
		//	t += ds;
		//	continue;
		//}

		//transmittance = exp(-absFac * (ds/5.0) * density);
		//transmittanceAll *= transmittance;

		//incidentLight = IncidentLighting(pos, false);
		//colour += (1.0 - transmittance) * incidentLight * transmittanceAll / absFac;

		//opacity += (1.0 - dot(transmittance, 1.0.xxx) / 3.0)*(1.0 - opacity);
		//t += ds/5.0;

		//if (opacity > 0.99)
		//{
		//	opacity = 1.0;
		//	break;
		//}
	}
	return float4(colour, opacity);
}

float4 main(PixelInputType pin) : SV_TARGET
{
	viewRay = normalize(pin.Ray);
float3 posRelSun = bCameraPos / 1000.0 + float3(0.0, Rg, 0.0); // TODO: handling arbitrary position of Earth

															   // length from core
float r = length(posRelSun);
// cos zenith angle
float mu = dot(posRelSun, viewRay) / r;
float smu = sign(mu);

// TODO: better distance calculation; these are all over the place when grazing angle

// distance to ground along viewing ray
float tGround = -r * mu + smu * sqrt(r * r * (mu * mu - 1.0f) + Rg * Rg);

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
if (tCloudTop < 0.0) return 0.0.xxxx;

// don't add what's behind the camera
tCloudBase = max(tCloudBase, 0.0);

if (tGround > 0.0)
tCloudTop = min(tCloudTop, tGround);

//clip(tCloudTop > tCloudBase ? 1.0 : -1.0);
if (tCloudTop < tCloudBase) return 0.0.xxxx;

// set global sun direction vector
bSunDir1 = -normalize(bSunDir);
cosViewSun = dot(viewRay, bSunDir1);

float4 clouds = MarchClouds(posRelSun, viewRay, tCloudBase, tCloudTop);

clouds.rgb = HDR(clouds.rgb);

return clouds;

/*
float3 baseCloudPos = float3(0.0, Rg + 0.01, 0.0);//bCameraPos;
bSunDir1 = -normalize(bSunDir);
v = bSunDir1;

float r = length(x);
float mu = dot(x, v) / r;
float tg = -r * mu + sqrt(r * r * (mu * mu - 1.0f) + Rg * Rg);

//clip(isnan(tg) || tg >= 0 ? -1 : 1);

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

sunLight = sun(x, tg, bSunDir1, bSunDir1, r, dot(bSunDir1, x));

v = normalize(pin.Ray);

x += bCameraPos;

float mu = dot(x, viewRay) / r;
float tCloudBase = -r * mu + sqrt(r * r * (mu * mu - 1.0f) + Rl * Rl);
float tCloudTop = -r * mu + sqrt(r * r * (mu * mu - 1.0f) + Rh * Rh);

tg = -r * mu + sqrt(r * r * (mu * mu - 1.0f) + Rg * Rg);

if (tg > 0.0)
{
return 0.0.xxxx;
}

x = bCameraPos;
float undersqrt = pow(dot(v, x), 2.0) - dot(x, x) + 100.0;

float d0 = -dot(v, x) - sqrt(undersqrt);
float d1 = -dot(v, x) + sqrt(undersqrt);

return MarchCloud(x, v, t0, t1);*/
}




















































//
//static float3 noise_kernel[] =
//{
//	float3(-0.393554, 0.0388882, 0.918478),
//	float3(-0.497004, -0.80077, -0.334296),
//	float3(0.676772, 0.735482, 0.0323357),
//	float3(0.860109, -0.484374, -0.15998),
//	float3(-0.301109, -0.924295, -0.234545),
//	float3(0.789508, -0.480101, 0.382335),
//	float3(0.756813, -0.585189, -0.291186)
//};
//
//static float3 weather_data;
//
//cbuffer cbPerFramePS
//{
//	float3 bCameraPos;
//	float bExposure;
//	float3 bSunDir;
//	float2 pad;
//};
//
//struct PixelInputType
//{
//	float4 PosH : SV_POSITION;
//	float3 Ray : TEXCOORD;
//};
//
//// direct sun light for ray x+tv, when sun in direction s (=L0)
//float3 sun(float3 x, float t, float3 v, float3 s, float r, float mu)
//{
//	if (t > 0.0)
//		return float3(0.0, 0.0, 0.0);
//	else
//	{
//		float3 transmittance = r <= Rt ? getTransmittanceWithShadow(r, mu) : float3(1.0, 1.0, 1.0); // T(x,x0)
//		float isun = step(cos(PI / 180.0), dot(v, s)) * ISun; // Lsun dot(s, v);// 
//		return transmittance*isun; // Eq (9)
//	}
//}
//
//float ComputeScattering(float lightDotView)
//{
//	float result = 1.0f - G_SCATTERING * G_SCATTERING;
//	result /= (4.0f * PI * pow(1.0f + G_SCATTERING * G_SCATTERING - (2.0f * G_SCATTERING) * lightDotView, 1.5f));
//	return result;
//}
//
//float GetNormalizedHeight(float3 inPosition, float2 inCloudMinMax)
//{
//	return saturate((inPosition.y - inCloudMinMax.x - Rg) / (inCloudMinMax.y - inCloudMinMax.x));
//}
//
//// Utility function that maps a value from one range to another
//float Remap(float original_value, float original_min, float original_max, float new_min, float new_max)
//{
//	return new_min + (((clamp(original_value, original_min, original_max) - original_min) /
//		(original_max - original_min)) * (new_max - new_min));
//}
//
//float GetDensityHeightGradient(float3 p, float3 weather_data, float2 inCloudMinMax = float2(1.5, 4.0))
//{
//	float normalized_height = 1.0 - GetNormalizedHeight(p, inCloudMinMax);
//	float density = cloudsType.SampleLevel(samTrilinearSam, float2(weather_data.b, normalized_height), 0).r;
//	// TODO: more dependent on the weather
//	return density;
//}
//
//float SampleCloudDensity(float3 p, float3 weather_data, uint mip_level, bool skip_details)
//{
//	// Read the low-frequency Perlin-Worley
//	float base_cloud = cloudsGeneral.SampleLevel(samTrilinearSam, p, mip_level).r;
//
//	// Get the density-height gradient using the density height function
//	float density_height_gradient = GetDensityHeightGradient(p, weather_data);
//
//	// Apply the height function to the base cloud shape
//	base_cloud *= density_height_gradient;
//
//	// CLoud coverage is store in weather_data's red channel
//	float cloud_coverage = weather_data.r;
//
//	// Use remap to apply the cloud coverage attribute
//	//float base_cloud_with_coverage = Remap(base_cloud, cloud_coverage, 1.0, 0.0, 1.0);
//	float base_cloud_with_coverage = smoothstep(1.0 - cloud_coverage, 1.0, base_cloud);
//
//	// Multiply the result by the cloud coverage
//	base_cloud_with_coverage *= cloud_coverage;
//
//	return base_cloud_with_coverage;
//}
//
//// Henyey-Greenstein phase function
//float HenyeyGreenstein(float cos_theta, float inG)
//{
//	return ((1.0 - inG * inG) / pow((1.0 + inG * inG - 2.0 * inG * cos_theta), 3.0 / 2.0)) / (4.0 * 3.14159);
//}
//
//// Energy at given point in cloud
//float Energy(float density, float theta, float rain_multiplier, float g)
//{
//	return 2.0*exp(-density * rain_multiplier)*(1.0 - exp(-2.0*density)*HenyeyGreenstein(theta, g));
//}
//
//float SampleCloudDensityAlongRay(float3 p)
//{
//	return 0.0;
//}
//
//// A function to gather density in a cone for use with lighting clouds
//float SampleCloudDensityAlongCone(float3 p, float3 ray_direction)
//{
//	float density_along_cone = 0.0;
//	float density_along_view_cone;
//	float weather_data;
//	float mip_level;
//
//	float light_step;
//
//	// How wide to make the cone
//	float cone_spread_multiplier = length(light_step);
//
//	// Lighting ray-march loop
//	for (uint i = 0; i <= 6; ++i)
//	{
//		// Add the current step offset to the sample position
//		p += light_step + (cone_spread_multiplier * noise_kernel[i] * float(i));
//
//		if (density_along_view_cone < 0.3)
//		{
//			// Sample cloud density the expensive way
//			density_along_cone += SampleCloudDensity(p, weather_data, mip_level + 1, false);
//		}
//		else
//		{
//			// Sample cloud density the cheap way, using only one lecel of noise
//			density_along_cone += SampleCloudDensity(p, weather_data, mip_level + 1, true);
//		}
//	}
//}
//
//float RayMarchCloud(float3 y0, float3 y1)
//{
//	float3 p = y0;
//	float3 step = (y1 - y0) / 64.0;
//	uint sample_count = 64;
//	weather_data.x = 0.2;
//
//	float density = 0.0;
//	float cloud_test = 0.0;
//	uint zero_density_sample_count = 0;
//
//	// Start the main ray-march loop
//	for (uint i = 0; i < sample_count; ++i)
//	{
//		// cloud_test starts as zeto so we always ecaluate the second case from the beginning
//		if (cloud_test > 0.0)
//		{
//			// Sample density the expensive way by setting the last parameter to false, indicating a full sample
//			float sampled_density = SampleCloudDensity(p, weather_data, 0, false);
//
//			// If we just samples a zeto, increment the counter
//			if (sampled_density == 0.0)
//			{
//				++zero_density_sample_count;
//			}
//			// If we are doing an expensive sample that is still potentially in the cloud
//			if (zero_density_sample_count != 6)
//			{
//				density += sampled_density;
//				if (sampled_density != 0.0)
//				{
//					// SampleCloudDensityAlongRay just walks in the given direction from start point
//					// and takes X number of lightning sample
//					//density_along_light_ray = SampleCloudDensityAlongRay(p);
//				}
//				p += step;
//			}
//			// If not, then set cloud_test to zero so that we go back to the cheap sample case
//			else
//			{
//				cloud_test = 0.0;
//				zero_density_sample_count = 0;
//			}
//		}
//		else
//		{
//			// Sample density the cheap way, only using the low-frequency noise
//			cloud_test = SampleCloudDensity(p, weather_data, 0/*mip_level*/, true);
//			if (cloud_test == 0.0)
//			{
//				p += step;
//			}
//		}
//	}
//
//	return density;
//}
//
//float sampleCheap(float3 x)
//{
//	float4 clouds = cloudsGeneral.SampleLevel(samTrilinearSam, float3(x.x/4, x.y, x.z/4), 0);
//	float4 weather = weatherPar.SampleLevel(samTrilinearSam, (x.xz + float2(70.0f, 70.0f))/140.0f, 0);
//	float4 type = cloudsType.SampleLevel(samTrilinearSam, float2(weather.z, 1-x.y), 0);
//
//	float4 base = clouds * type.x;
//	float4 cutoff = step(1.0f - weather.x, base);
//
//	float4 alpha = cutoff * base * weather.x * (0.625f * x.y + 0.375f);
//
//	return alpha.x;
//}
//
//float2 rayMarchAlpha(float3 y0, float3 y1, float mu)
//{
//	int steps = -64 * mu + 128;
//	float2 transmittance = float2(1.0f, 0.0f);
//	
//	uint zero = 0;
//	float3 v = (y1 - y0) / 16.0f;
//	float step = 1.0f;
//	float da = 0.0f;
//	float i = 0.0f;
//
//	float3 weather = weatherPar.SampleLevel(samTrilinearSam, (y0.xz + float2(70.0f, 70.0f)) / 140.0f, 0).rgb;
//
//	for (i = 0.0f; i < 16; i += step)
//	{
//		da = 0.0f;
//		if (transmittance.x < 0.7)
//		{
//			da = SampleCloudDensity(y0 + i*v, weather, 0, true);
//			transmittance.x *= exp(-step*da);
//			transmittance.y += exp(-step*da)*(1 - exp(-i * 2));
//			step = 1.0f;
//		}
//		else
//		{
//			if (step > 0.5f)
//			{
//				da = sampleCheap(y0 + i*v);
//				if (da > 0.0f)
//				{
//					i = max(i - step, 0.0f);
//					step = 0.1f;
//				}
//			}
//			else if (zero < 3)
//			{
//				da = sampleCheap(y0 + i*v);
//				if (da < 1e-9) ++zero;
//				else
//				{
//					zero = 0;
//					transmittance.x *= exp(-step*da);
//					transmittance.y += exp(-step*da)*(1 - exp(-i * 2));
//				}
//			}
//			else
//			{
//				da = sampleCheap(y0 + i*v);
//				transmittance.x *= exp(-step*da);
//				transmittance.y += exp(-step*da)*(1 - exp(-i * 2));
//				step = 1.0f;
//			}
//		}
//		if (transmittance.x <= 0.01f) break;
//	}
//
//	return transmittance;
//}
//
//float4 main(PixelInputType pin) : SV_TARGET
//{
//	float3 v = normalize(pin.Ray);
//	float3 x = bCameraPos;
//	float3 bSunDir1 = normalize(bSunDir);
//
//	float r = length(x);
//	float mu = dot(x, v) / r;
//	float t0 = -r * mu + sqrt(r * r * (mu * mu - 1.0f) + Rl * Rl);
//	float t1 = -r * mu + sqrt(r * r * (mu * mu - 1.0f) + Rh * Rh);
//	float tg = -r * mu + sqrt(r * r * (mu * mu - 1.0f) + Rg * Rg);
//
//	clip(isnan(tg) || tg >= 0 ? -1 : 1);
//
//	float3 g = x - float3(0.0, 0.0, Rg + 10.0);
//	float a = v.x * v.x + v.y * v.y - v.z * v.z;
//	float b = 2.0 * (g.x * v.x + g.y * v.y - g.z * v.z);
//	float c = g.x *g.x + g.y * g.y - g.z * g.z;
//	float d = -(b + sqrt(b * b - 4.0 * a * c)) / (2.0 * a);
//	bool cone = d > 0.0 && abs(x.z + d * v.z - Rg) <= 10.0;
//
//	if (tg > 0.0)
//	{
//		if (cone && d < tg)
//			tg = d;
//	}
//	else if (cone)
//		tg = d;
//	
//	float3 sunColour = sun(x, tg, bSunDir1, bSunDir1, r, dot(bSunDir1, x)); // L0
//
//	float3 y0 = x + t0*v;
//	float3 y1 = x + t1*v;
//
//	weather_data = weatherPar.SampleLevel(samTrilinearSam, (y0.xz + float2(70.0f, 70.0f)) / 140.0f, 0).rgb;
//
//	//float2 alpha = rayMarchAlpha(y0, y1, mu);
//	float ret = RayMarchCloud(y0, y1);
//	return float4(float3(0.5, 0.5, 0.5), ret);
//}