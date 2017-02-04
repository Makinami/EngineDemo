#include "Common.hlsli"

RWTexture2D<float3> transmittance : register(u0);

// NOTE: check if pushing horizon check to the end will be faster
float opticalDepth(float H, float alt, float vzAngle)
{
	// if ray below horizon return max density
	if (vzAngle < -sqrt(1.0f - ((groundR * groundR) / (alt * alt))))
		return 1e9;

	float totalDepth = 0.0;
	float dx = intersectAtmosphereBoundry(alt, vzAngle) / float(TRANSMITTANCE_INTEGRAL_SAMPLES);

	float xi = 0.0;
	float yj = exp(-(alt - groundR) / H);

	for (uint i = 1; i <= TRANSMITTANCE_INTEGRAL_SAMPLES; ++i)
	{
		xi = float(i) * dx;
		float yi = exp(-(sqrt(alt*alt + xi*xi + 2.0*xi*alt*vzAngle) - groundR) / H);
		totalDepth += (yi + yj) / 2.0 * dx;
		yj = yi;
	}

	return totalDepth;
}

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	float alt, vzAngle;
	getTransmittanceAltVzAngle(DTid.xy, alt, vzAngle);
	float3 depth = betaR*opticalDepth(HR, alt, vzAngle) + betaMEx*opticalDepth(HM, alt, vzAngle);
	transmittance[DTid.xy] = exp(-depth);
}