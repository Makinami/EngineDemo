#include "Structures.hlsli"

/*
	PHISICAL MODEL PARAMETERS
*/
static const float AVERAGE_GROUND_REFLECTANCE = 0.1;

// Rayleigh
static const float HR = 8.0f;
static const float3 betaR = float3(5.8e-3, 1.35e-2, 3.31e-2);

// Mie
// default
static const float HM = 1.2;
static const float3 betaMSca = float3(4e-3, 4e-3, 4e-3);
static const float3 betaMEx = betaMSca / 0.9f;
static const float mieG = 0.8;
// clear
//static const float HM = 1.2;
//static const float3 betaMSca = float3(20e-3, 20e-3, 20e-3);
//static const float3 betaMEx = betaMSca / 0.9f;
//static const float mieG = 0.76;
// cloudy
//static const float HM = 3.0;
//static const float3 betaMSca = float3(3e-3, 3e-3, 3e-3);
//static const float3 betaMEx = betaMSca / 0.9f;
//static const float mieG = 0.65;


/*
	NUMERICAL INTEGRATION PARAMETERS
*/
static const uint TRANSMITTANCE_INTEGRAL_SAMPLES = 500;
static const uint INSCATTER_INTEGRAL_SAMPLES = 50;
static const uint IRRADIANCE_INTEGRAL_SAMPLES = 32;
static const uint INSCATTER_SPHERICAL_INTEGRAL_SAMPLES = 16;

static const float PI = 3.141592657f;


/*
	PARAMETRIZATION FUNCTIONS
*/
float2 getTransmittanceUV(float alt, float vzAngle)
{
	float uAlt, uVzAngle;

	uAlt = sqrt((alt - groundR) / (topR - groundR));
	uVzAngle = atan((vzAngle + 0.15) / 1.15 * tan(1.5)) / 1.5;

	return float2(uVzAngle, uAlt);
}

void getTransmittanceAltVzAngle(float2 pos, out float alt, out float vzAngle)
{
	alt = (0.5 + pos.y) / (float)TRANSMITTANCE_H;
	vzAngle = (0.5 + pos.x) / (float)TRANSMITTANCE_W;

	alt = groundR + (alt*alt)*(topR - groundR);
	vzAngle = -0.15 + tan(1.5*vzAngle)/tan(1.5) * 1.15;
}

float2 getIrradianceUV(float alt, float szAngle)
{
	return float2((szAngle + 0.2) / 1.2, (alt - groundR) / (topR - groundR));
}

void getIrradianceAltSzAngle(int2 pos, out float alt, out float szAngle)
{
	alt = groundR + pos.y / (float(SKY_H) - 1.0) * (topR - groundR);
	szAngle = -0.2f + pos.x / (float(SKY_W) - 1.0) * 1.2;
}

void getTexture4DUVW(float alt, float vzAngle, float szAngle, float vsAngle,
					 out float uAlt, out float uVzAngle, out float uSzAngle, out float uVsAngle, out float lerp)
{
	float H = sqrt(topR*topR - groundR*groundR);
	float rho = sqrt(alt*alt - groundR*groundR);

	float rmu = alt*vzAngle;
	float delta = rmu * rmu - alt*alt + groundR*groundR;
	float4 cst = (rmu < 0.0 && delta > 0.0) ? float4(1.0, 0.0, 0.0, 0.5 - 0.5 / float(RES_VZ)) : float4(-1.0, H*H, H, 0.5 + 0.5 / float(RES_VZ));
	uAlt = 0.5 / float(RES_ALT) + rho / H * (1.0 - 1.0 / float(RES_ALT));
	uVzAngle = cst.w + (rmu * cst.x + sqrt(delta + cst.y)) / (rho + cst.z) * (0.5 - 1.0 / float(RES_VZ));
	uSzAngle = 0.5 / float(RES_SZ) + (atan(max(szAngle, -0.1975) * tan(1.26 * 1.1)) / 1.1 + 0.74) * 0.5 * (1.0 - 1.0 / float(RES_SZ));

	lerp = (vsAngle + 1.0) / 2.0 * (float(RES_VS) - 1.0);
	uVsAngle = floor(lerp);
	lerp = lerp - uVsAngle;
}

void getVzSzVsAngles(float2 pos, float alt, float4 dhdH, out float vzAngle, out float szAngle, out float vsAngle)
{
	float x = pos.x;
	float y = pos.y;

	if (y < float(RES_VZ) / 2.0)
	{
		float d = 1.0 - y / (float(RES_VZ) / 2.0 - 1.0);
		d = min(max(dhdH.z, d*dhdH.w), dhdH.w*0.999);
		vzAngle = (groundR*groundR - alt*alt - d*d) / (2.0*alt*d);
		vzAngle = min(vzAngle, -sqrt(1.0 - (groundR*groundR) / (alt*alt)) - 0.001);
	}
	else
	{
		float d = (y - float(RES_VZ) / 2.0) / (float(RES_VZ) / 2.0 - 1.0);
		d = min(max(dhdH.x, d*dhdH.y), dhdH.y*0.999);
		vzAngle = (topR*topR - alt*alt - d*d) / (2.0*d*alt);
	}
	szAngle = fmod(x, float(RES_SZ)) / (float(RES_SZ) - 1.0);
	szAngle = tan((2.0*szAngle - 1.0 + 0.26)*1.1) / tan(1.26 * 1.1);
	vsAngle = -1.0 + floor(x / float(RES_SZ)) / (float(RES_VS) - 1.0) * 2.0;
}


/*
	UTILITY FUNCTIONS
*/
// nearest intersection of ray alt, vzAngle with ground or top atmoshpere boundry
float intersectAtmosphereBoundry(in float alt, in float vzAngle)
{
	float dout = -alt * vzAngle + sqrt(alt * alt * (vzAngle * vzAngle - 1.0) + topR * topR);
	float deltaSq = alt * alt * (vzAngle * vzAngle - 1.0) + groundR * groundR;
	if (deltaSq >= 0.0)
	{
		float din = -alt * vzAngle - sqrt(deltaSq);
		if (din >= 0.0)
			dout = min(dout, din);
	}
	return dout;
}

#ifdef USE_TRANSMITTANCE
float3 getTransmittance(float alt, float vzAngle)
{
	return transmittance.SampleLevel(samBilinearClamp, getTransmittanceUV(alt, vzAngle), 0).rgb;
}

float3 getTransmittance(float alt, float vzAngle, float dist)
{
	float alt1 = sqrt(alt*alt + dist*dist + 2.0*alt*vzAngle*dist);
	float vzAngle1 = (alt*vzAngle + dist) / alt1;
	if (vzAngle > 0.0)
		return min(getTransmittance(alt, vzAngle) / getTransmittance(alt1, vzAngle1), 1.0);
	else
		return min(getTransmittance(alt1, -vzAngle1) / getTransmittance(alt, -vzAngle), 1.0);
}
#endif

#ifdef USE_IRRADIANCE
float4 getIrradiance(float alt, float szAngle)
{
	return deltaE.SampleLevel(samBilinearClamp, getIrradianceUV(alt, szAngle), 0);
}
#endif

#ifdef USE_DELTAS
float4 getDeltaSR(float alt, float vzAngle, float szAngle, float vsAngle)
{
	float uAlt, uVzAngle, uSzAngle, uVsAngle, lerp;
	getTexture4DUVW(alt, vzAngle, szAngle, vsAngle, uAlt, uVzAngle, uSzAngle, uVsAngle, lerp);
	return deltaSR.SampleLevel(samBilinearClamp, float3((uVsAngle + uSzAngle) / float(RES_VS), uVzAngle, uAlt), 0) * (1.0 - lerp) +
		deltaSR.SampleLevel(samBilinearClamp, float3((uVsAngle + uSzAngle + 1.0) / float(RES_VS), uVzAngle, uAlt), 0) * lerp;
}

float4 getDeltaSM(float alt, float vzAngle, float szAngle, float vsAngle)
{
	float uAlt, uVzAngle, uSzAngle, uVsAngle, lerp;
	getTexture4DUVW(alt, vzAngle, szAngle, vsAngle, uAlt, uVzAngle, uSzAngle, uVsAngle, lerp);
	return deltaSM.SampleLevel(samBilinearClamp, float3((uVsAngle + uSzAngle) / float(RES_VS), uVzAngle, uAlt), 0) * (1.0 - lerp) +
		deltaSM.SampleLevel(samBilinearClamp, float3((uVsAngle + uSzAngle + 1.0) / float(RES_VS), uVzAngle, uAlt), 0) * lerp;
}
#endif

#ifdef USE_DELTAJ
float4 getDeltaJ(float alt, float vzAngle, float szAngle, float vsAngle)
{
	float uAlt, uVzAngle, uSzAngle, uVsAngle, lerp;
	getTexture4DUVW(alt, vzAngle, szAngle, vsAngle, uAlt, uVzAngle, uSzAngle, uVsAngle, lerp);
	return deltaJ.SampleLevel(samBilinearClamp, float3((uVsAngle + uSzAngle) / float(RES_VS), uVzAngle, uAlt), 0) * (1.0 - lerp) +
		deltaJ.SampleLevel(samBilinearClamp, float3((uVsAngle + uSzAngle + 1.0) / float(RES_VS), uVzAngle, uAlt), 0) * lerp;
}
#endif

// Raylight phase function
float phaseFunctionR(float mu)
{
	return (3.0 / (16.0 * PI)) * (1.0 + mu*mu);
}

// Mie phase function
float phaseFunctionM(float mu) {
	return 1.5 * 1.0 / (4.0 * PI) * (1.0 - mieG*mieG) * pow(1.0 + (mieG*mieG) - 2.0*mieG*mu, -3.0 / 2.0) * (1.0 + mu * mu) / (2.0 + mieG*mieG);
}

// approximated single Mie scattering (cf. approximate Cm in paragraph "Angular precision")
float3 getMie(float4 rayMie) { // rayMie.rgb=C*, rayMie.w=Cm,r
	return rayMie.rgb * rayMie.w / max(rayMie.r, 1e-4) * (betaR.r / betaR);
}
