// common.hlsli

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

static const int TRANSMITTANCE_INTEGRAL_SAMPLES = 500;
static const int INSCATTER_INTEGRAL_SAMPLES = 50;
static const int IRRADIANCE_INTEGRAL_SAMPLES = 32;
static const int INSCATTER_SPHERICAL_INTEGRAL_SAMPLES = 16;

static const float PI = 3.141592657f;


/*
	PARAMETERIZATION OPTIONS
*/

#define TRANSMITTANCE_NON_LINEAR
#define INSCATTER_NON_LINEAR


/*
	PARAMETERIZATION FUNCTIONS
*/

float2 getTransmittanceUV(float r, float mu)
{
	float uR, uMu;
#ifdef TRANSMITTANCE_NON_LINEAR
	uR = sqrt((r - Rg) / (Rt - Rg));
	uMu = atan((mu + 0.15) / (1.0 + 0.15)*tan(1.5)) / 1.5;
#else
	uR = (r - Rg) / (Rt - Rg);
	uMu = (mu + 0.15) / (1.0 + 0.15);
#endif
	return float2(uMu, uR);
}

void getTransmittanceRMu(int2 pos, out float r, out float muS)
{
	r = (0.5f + (float)(pos.y)) / float(TRANSMITTANCE_H);
	muS = (0.5f + (float)(pos.x)) / float(TRANSMITTANCE_W);
#ifdef TRANSMITTANCE_NON_LINEAR
	r = Rg + (r*r)*(Rt - Rg);
	muS = -0.15 + tan(1.5*muS) / tan(1.5)*(1.0 + 0.15);
#else
	r = Rg + r*(Rt - Rg);
	muS = -0.15 + muS*(1.0 + 0.15);
#endif
}

float2 getIrradianceUV(float r, float muS)
{
	return float2((muS + 0.2) / (1.0 + 0.2), (r - Rg) / (Rt - Rg));
}

void getIrradianceRMuS(int2 pos, out float r, out float muS)
{
	r = Rg + pos.y / (float(SKY_H) - 1.0f)*(Rt - Rg);
	muS = -0.2f + pos.x / (float(SKY_W) - 1.0f)*(1.0f + 0.2f);
}

void getTexture4DUVW(float r, float mu, float muS, float nu,
	out float uR, out float uMu, out float uMuS, out float uNu, out float lerp)
{
	float H = sqrt(Rt*Rt - Rg*Rg);
	float rho = sqrt(r*r - Rg*Rg);
#ifdef INSCATTER_NON_LINEAR
	float rmu = r*mu;
	float delta = rmu * rmu - r * r + Rg * Rg;
	float4 cst = (rmu < 0.0 && delta > 0.0) ? float4(1.0, 0.0, 0.0, 0.5 - 0.5 / float(RES_MU)) : float4(-1.0, H * H, H, 0.5 + 0.5 / float(RES_MU));
	uR = 0.5 / float(RES_R) + rho / H * (1.0 - 1.0 / float(RES_R));
	uMu = cst.w + (rmu * cst.x + sqrt(delta + cst.y)) / (rho + cst.z) * (0.5 - 1.0 / float(RES_MU));
	uMuS = 0.5 / float(RES_MU_S) + (atan(max(muS, -0.1975) * tan(1.26 * 1.1)) / 1.1 + (1.0 - 0.26)) * 0.5 * (1.0 - 1.0 / float(RES_MU_S));
#else
	uR = 0.5 / float(RES_R) + rho / H * (1.0 - 1.0 / float(RES_R));
	uMu = 0.5 / float(RES_MU) + (mu + 1.0) / 2.0 * (1.0 - 1.0 / float(RES_MU));
	uMuS = 0.5 / float(RES_MU_S) + max(muS + 0.2, 0.0) / 1.2 * (1.0 - 1.0 / float(RES_MU_S));
#endif
	lerp = (nu + 1.0) / 2.0 * (float(RES_NU) - 1.0);
	uNu = floor(lerp);
	lerp = lerp - uNu;
}

void getMuMuSNu(float2 pos, float r, float4 dhdH, out float mu, out float muS, out float nu)
{
	float x = pos.x;
	float y = pos.y;
#ifdef INSCATTER_NON_LINEAR
	if (y < float(RES_MU) / 2.0)
	{
		float d = 1.0 - y / (float(RES_MU) / 2.0 - 1.0);
		d = min(max(dhdH.z, d*dhdH.w), dhdH.w*0.999);
		mu = (Rg*Rg - r*r - d*d) / (2.0*r*d);
		mu = min(mu, -sqrt(1.0 - (Rg / r)*(Rg / r)) - 0.001);
	}
	else
	{
		float d = (y - float(RES_MU) / 2.0) / (float(RES_MU) / 2.0 - 1.0);
		d = min(max(dhdH.x, d*dhdH.y), dhdH.y*0.999);
		mu = (Rt*Rt - r*r - d*d) / (2.0*r*d);
	}
	muS = fmod(x, float(RES_MU_S)) / (float(RES_MU_S) - 1.0);
	muS = tan((2.0*muS - 1.0 + 0.26)*1.1) / tan(1.26*1.1);
	nu = -1.0 + floor(x / float(RES_MU_S)) / (float(RES_NU) - 1.0) * 2.0;
#else
	mu = -1.0 + 2.0*y / (float(RES_MU) - 1.0);
	muS = fmod(x, float(RES_MU_S)) / (float(RES_MU_S) - 1.0);
	muS = -0.2 + muS*1.2;
	nu = -1.0 + floor(x / float(RES_MU_S)) / (float(RES_NU) - 1.0)*2.0;
#endif
}


/*
	UTINILY FUNCTIONS
*/

// nearest intersection of ray r, mu with ground or top atmoshpere boundry
// mu = cos(ray zenith angle at ray origin)
float limit(float r, float mu)
{
	float dout = -r*mu + sqrt(r*r*(mu*mu - 1) + RL*RL);  // TODO: RL - strange
	float delta2 = r*r*(mu*mu - 1.0) + Rg*Rg;
	if (delta2 >= 0.0)
	{
		float din = -r*mu - sqrt(delta2);
		if (din >= 0.0)
		{
			dout = min(dout, din);
		}
	}
	return dout;
}

#ifdef USE_TRANSMITTANCE
float3 getTransmittance(float r, float mu)
{
	return transmittance.SampleLevel(samTransmittance, getTransmittanceUV(r, mu), 0).rgb;
}

// transmittance  of atmosphere between x and x0
// assume segment x,x0 not intersecting ground
// r =||x||, mu = cos(zenith angleof [x,x0) ray at x), v = unit direcrionof [x,x0)
float3 getTransmittance(float r, float mu, float3 v, float3 x0)
{
	float r1 = length(x0);
	float mu1 = dot(x0, v) / r;
	if (mu > 0.0)
		return min(getTransmittance(r, mu) / getTransmittance(r1, mu1), 1.0);
	else
		return min(getTransmittance(r1, -mu1) / getTransmittance(r, -mu), 1.0);
}

float3 getTransmittanceWithShadow(float r, float mu)
{
	return mu < -sqrt(1.0 - (Rg / r)*(Rg / r)) ? float3(0.0, 0.0, 0.0) : getTransmittance(r, mu);
}

float3 getTransmittance(float r, float mu, float d)
{
	float r1 = sqrt(r*r + d*d + 2.0 * r * mu * d);
	float mu1 = (r*mu + d) / r1;
	if (mu > 0.0)
		return min(getTransmittance(r, mu) / getTransmittance(r1, mu1), 1.0);
	else
		return min(getTransmittance(r1, -mu1) / getTransmittance(r, -mu), 1.0);
}
#endif

// optical depth for ray (r,mu) o flength d, using analytic formula
// (mu = cos(view zenith angle)), intersections with groundignored
// H = height scale of exponential density function
float opticalDepth(float H, float r, float mu, float d)
{
	float a = sqrt((0.5 / H)*r);
	float2 a01 = a*float2(mu, mu + d / r);
	float2 a01s = sign(a01);
	float2 a01sq = a01*a01;
	float x = a01s.y > a01s.x ? exp(a01sq.x) : 0.0;
	float2 y = a01s / (2.3193*abs(a01) + sqrt(1.52*a01sq + 4.0))*float2(1.0, exp(-d / H*(d / (2.0*r) + mu)));
	return sqrt((6.2831*H)*r)*exp((Rg - r) / H)*(x + dot(y, float2(1.0, -1.0)));
}

float3 analyticTransmittance(float r, float mu, float d)
{
	return exp(-betaR * opticalDepth(HR, r, mu, d) - betaMEx * opticalDepth(HM, r, mu, d));
}

#ifdef USE_IRRADIANCE
float4 getIrradiance(float r, float muS)
{
	return deltaE.SampleLevel(samIrradiance, getIrradianceUV(r, muS), 0);
}
#endif

#ifdef USE_DELTAS
float4 getDeltaSR(float r, float mu, float muS, float nu)
{
	float uNu, uMuS, uMu, uR, lerp;
	getTexture4DUVW(r, mu, muS, nu, uR, uMu, uMuS, uNu, lerp);
	return deltaSR.SampleLevel(samDeltaSR, float3((uNu + uMuS) / float(RES_NU), uMu, uR), 0) * (1.0 - lerp) +
		deltaSR.SampleLevel(samDeltaSR, float3((uNu + uMuS + 1.0) / float(RES_NU), uMu, uR), 0) * lerp;
}

float4 getDeltaSM(float r, float mu, float muS, float nu)
{
	float uNu, uMuS, uMu, uR, lerp;
	getTexture4DUVW(r, mu, muS, nu, uR, uMu, uMuS, uNu, lerp);
	return deltaSM.SampleLevel(samDeltaSM, float3((uNu + uMuS) / float(RES_NU), uMu, uR), 0) * (1.0 - lerp) +
		deltaSM.SampleLevel(samDeltaSM, float3((uNu + uMuS + 1.0) / float(RES_NU), uMu, uR), 0) * lerp;
}
#endif

#ifdef USE_DELTAJ
float4 getDeltaJ(float r, float mu, float muS, float nu)
{
	float uNu, uMuS, uMu, uR, lerp;
	getTexture4DUVW(r, mu, muS, nu, uR, uMu, uMuS, uNu, lerp);
	return deltaJ.SampleLevel(samDeltaJ, float3((uNu + uMuS) / float(RES_NU), uMu, uR), 0) * (1.0 - lerp) +
		deltaJ.SampleLevel(samDeltaJ, float3((uNu + uMuS + 1.0) / float(RES_NU), uMu, uR), 0) * lerp;
}
#endif

#ifdef USE_INSCATTER
float4 getInscatter(float r, float mu, float muS, float nu)
{
	float uNu, uMuS, uMu, uR, lerp;
	getTexture4DUVW(r, mu, muS, nu, uR, uMu, uMuS, uNu, lerp);
	return inscatterTex.SampleLevel(samInscatter, float3((uNu + uMuS) / float(RES_NU), uMu, uR), 0) * (1.0 - lerp) +
		inscatterTex.SampleLevel(samInscatter, float3((uNu + uMuS + 1.0) / float(RES_NU), uMu, uR), 0) * lerp;
}
#endif

// Rayleight phase function
float phaseFunctionR(float mu)
{
	return (3.0 / (16.0 * PI)) * (1.0 + mu*mu);
}

// Mie phase function
float phaseFunctionM(float mu)
{
	return 1.5 * 1.0 / (4.0 * PI) * (1.0 - mieG*mieG) * pow(1.0 + (mieG*mieG) - 2.0*mieG*mu, -3.0 / 2.0) * (1.0 + mu * mu) / (2.0 + mieG*mieG);
}

// approximated single Mie scattering
float3 getMie(float4 rayMie)
{
	return rayMie.rgb*rayMie.w / max(rayMie.r, 1e-4)*(betaR.r / betaR);
}