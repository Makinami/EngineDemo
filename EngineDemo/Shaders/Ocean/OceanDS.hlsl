#include "perFrameCB.hlsli" // b1

cbuffer constBuffer : register(b0)
{
	float4 INVERSE_GRID_SIZE;
	float4 GRID_SIZE;
};

Texture2DArray wavesDisplacement : register(t0);
SamplerState samAnisotropic : register(s1);

struct DomainOut
{
	float4 PosH  : SV_POSITION;
	float3 PosF : TEXTCOORD0;
	float3 PosW : TEXTCOORD1;
};

// Output control point
struct HullOut
{
	float3 PosF : POSITION;
};


// Output patch constant data.
struct HS_CONSTANT_DATA_OUTPUT
{
	float EdgeTessFactor[3]			: SV_TessFactor; // e.g. would be [4] for a quad domain
	float InsideTessFactor			: SV_InsideTessFactor; // e.g. would be Inside[2] for a quad domain
};

#define NUM_CONTROL_POINTS 3

[domain("tri")]
DomainOut main(
	HS_CONSTANT_DATA_OUTPUT input,
	float3 domain : SV_DomainLocation,
	const OutputPatch<HullOut, NUM_CONTROL_POINTS> patch)
{
	DomainOut dout;

	dout.PosF = float3(
		patch[0].PosF*domain.x + patch[1].PosF*domain.y + patch[2].PosF*domain.z);
	
	float dist = dout.PosF.z;

	float3 dP = float3(0.0, 0.0, 0.0);

	dP += wavesDisplacement.SampleLevel(samAnisotropic, float3(dout.PosF.xy / GRID_SIZE.x, 0), 0.0).rbg;
	dP += (1.0 - smoothstep(GRID_SIZE.x, 1.2*GRID_SIZE.x, dist))*wavesDisplacement.SampleLevel(samAnisotropic, float3(dout.PosF.xy / GRID_SIZE.y, 1), 0.0).rbg;
	dP += (1.0 - smoothstep(GRID_SIZE.y, 1.2*GRID_SIZE.y, dist))*wavesDisplacement.SampleLevel(samAnisotropic, float3(dout.PosF.xy / GRID_SIZE.z, 2), 0.0).rbg;
	dP += (1.0 - smoothstep(GRID_SIZE.z, 1.2*GRID_SIZE.z, dist))*wavesDisplacement.SampleLevel(samAnisotropic, float3(dout.PosF.xy / GRID_SIZE.w, 3), 0.0).rbg;
	
	// NOTE: ifs - 11.51, in loop - 11.46, noneinloop - 11.10-11.31, byhand - 11.07
	/*[unroll(4)]
	for (int i = 0; i < 4; ++i)
	dP += (1 - smoothstep(GRID_SIZE[i], GRID_SIZE[i]*1.2, dist)) * wavesDisplacement.SampleLevel(samAnisotropic, float3(dout.PosF / GRID_SIZE[i], i), 0.0).rbg;*/
	/*float4 factors = float4(1.0, 1.0 - smoothstep(GRID_SIZE.y, GRID_SIZE.y*1.2, dist), 1.0 - smoothstep(GRID_SIZE.z, GRID_SIZE.z*1.2, dist), 1.0 - smoothstep(GRID_SIZE.w, GRID_SIZE.w*1.2, dist));
	if (factors.w > 0.0)
	{
		dP += wavesDisplacement.SampleLevel(samAnisotropic, float3(dout.PosF / GRID_SIZE.x, 0), 0.0).rbg;
		dP += factors.y * wavesDisplacement.SampleLevel(samAnisotropic, float3(dout.PosF / GRID_SIZE.y, 1), 0.0).rbg;
		dP += factors.z * wavesDisplacement.SampleLevel(samAnisotropic, float3(dout.PosF / GRID_SIZE.z, 2), 0.0).rbg;
		dP += factors.w * wavesDisplacement.SampleLevel(samAnisotropic, float3(dout.PosF / GRID_SIZE.w, 3), 0.0).rbg;
	}
	else if (factors.z > 0.0)
	{
		dP += wavesDisplacement.SampleLevel(samAnisotropic, float3(dout.PosF / GRID_SIZE.x, 0), 0.0).rbg;
		dP += factors.y * wavesDisplacement.SampleLevel(samAnisotropic, float3(dout.PosF / GRID_SIZE.y, 1), 0.0).rbg;
		dP += factors.z * wavesDisplacement.SampleLevel(samAnisotropic, float3(dout.PosF / GRID_SIZE.z, 2), 0.0).rbg;
	}
	else if (factors.y > 0.0)
	{
		dP += wavesDisplacement.SampleLevel(samAnisotropic, float3(dout.PosF / GRID_SIZE.x, 0), 0.0).rbg;
		dP += factors.y * wavesDisplacement.SampleLevel(samAnisotropic, float3(dout.PosF / GRID_SIZE.y, 1), 0.0).rbg;
	}
	else
		dP += wavesDisplacement.SampleLevel(samAnisotropic, float3(dout.PosF / GRID_SIZE.x, 0), 0.0).rbg; */

	dout.PosW = float3(dout.PosF.x, 0.0, dout.PosF.y) + dP * float3(lambdaV, 1.0, lambdaV);
	dout.PosH = mul(float4(dout.PosW, 1.0f), worldToScreenMatrix);

	return dout;
}
