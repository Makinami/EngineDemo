#include <..\WaterBruneton\atmosphere.hlsli>

cbuffer SkyMap : register(c0)
{
	float3 sunDir;
	float pad;
};

struct VertexOut
{
	float4 PosH : SV_POSITION;
	float2 uv : TEXTCOORD0;
};

float4 main(VertexOut pin) : SV_TARGET
{
	float2 uv = pin.uv;
	float l = dot(uv, uv);
	float4 result = float4(0.0, 0.0, 0.0, 1.0);

	if (l <= 1.02f)
	{
		if (l > 1.0f)
		{
			uv = uv / l;
			l = 1.0 / l;
		}

		// inverse stereographic
		float3 rDir = float3(2 * uv.x, 1.0 - l, 2 * uv.y) / (1.0 + l);

		float3 extinction;
		result.rgb = skyRadiance(earthPos, rDir, sunDir, extinction);
	}
	else
	{
		// below horizon:
		// use average fresnel * average sky radiance

		const float avgFresnel = 0.17;
		result.rgb = skyIrradiance(earthPos.y, sunDir.y) / PI * avgFresnel;
	}

	return result;
}