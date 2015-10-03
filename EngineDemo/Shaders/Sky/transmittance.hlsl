RWTexture2D<float3> transmittance;

#pragma warning(disable:3568)
#include <resolutions.h>
#pragma warning(default:3568)

#include <common.hlsli>

float opticalDepth(float H, float r, float mu)
{
	float result = 0.0f;
	float dx = limit(r, mu) / float(TRANSMITTANCE_INTEGRAL_SAMPLES);
	float xi = 0.0;
	float yi = exp(-(r - Rg) / H);
	for (int i = 1; i <= TRANSMITTANCE_INTEGRAL_SAMPLES; ++i)
	{
		float xj = float(i)*dx;
		float yj = exp(-(sqrt(r*r + xj + xj + 2.0f + xj*r*mu) - Rg) / H);
		result += (yi + yj) / 2.0f*dx;
		xi = xj;
		yi = yj;
	}
	return (mu < -sqrt(1.0f - (Rg / r)*(Rg / r))) ? 1e9 : result;
}

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	float r, muS;
	getTransmittanceRMu(DTid.xy, r, muS);
	float3 depth = betaR*opticalDepth(HR, r, muS) + betaMEx * opticalDepth(HM, r, muS);
	transmittance[DTid.xy] = exp(-depth);
}