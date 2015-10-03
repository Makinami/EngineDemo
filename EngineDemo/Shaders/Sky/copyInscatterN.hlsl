Texture3D<float4> inscatter : register(t0);
Texture3D<float4> deltaS : register(t1);

RWTexture3D<float4> copyInscatter : register(u0);

#pragma warning(disable:3568)
#include <resolutions.h>
#pragma warning(default:3568)

#include <common.hlsli>

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	float r = DTid.z / (RES_R - 1.0f);
	r = r * r;
	r = sqrt(Rg*Rg + r*(Rt*Rt - Rg*Rg)) + (DTid.z == 0 ? 0.01 : (DTid.z == RES_R - 1 ? -0.01 : 0.0));
	float4 dhdH = float4(Rt - r, sqrt(r*r - Rg*Rg) + sqrt(Rt*Rt - Rg*Rg), r - Rg, sqrt(r*r - Rg*Rg));

	float mu, muS, nu;
	getMuMuSNu(DTid.xy, r, dhdH, mu, muS, nu);

	copyInscatter[DTid] = inscatter[DTid] + float4(deltaS[DTid].rgb / phaseFunctionR(nu), 0.0);
}