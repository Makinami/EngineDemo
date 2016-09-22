Texture3D<float4> inscatter : register(t0);
Texture3D<float4> deltaS : register(t1);
RWTexture3D<float4> inscatterCopy : register(u0);

SamplerState samBilinearClamp : register(s0);

#include "Common.hlsli"

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	float alt = DTid.z / (RES_ALT - 1.0);
	alt = alt * alt;
	alt = sqrt(groundR*groundR + alt*(topR*topR - groundR*groundR)) + (DTid.z == 0 ? 0.01 : (DTid.z == RES_ALT - 1 ? -0.01 : 0.0));
	float4 dhdH = float4(topR - alt, sqrt(alt*alt - groundR*groundR) + sqrt(topR*topR - groundR*groundR), alt - groundR, sqrt(alt*alt - groundR*groundR));

	float vzAngle, szAngle, vsAngle;
	getVzSzVsAngles(DTid.xy, alt, dhdH, vzAngle, szAngle, vsAngle);

	// WTF?! Should be inscatter + deltaS/pFR, so why it's working this way?
	inscatterCopy[DTid] = deltaS[DTid] + float4(inscatter[DTid].rgb / phaseFunctionR(vsAngle), 0.0);
}