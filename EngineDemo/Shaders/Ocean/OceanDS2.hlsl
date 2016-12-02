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
	float4 params : TEXTCOORD2;
};

// Output control point
struct HullOut
{
	float2 Pos : POSITION;
	float4 param : TEXTCOORD;
};


// Output patch constant data.
struct HS_CONSTANT_DATA_OUTPUT
{
	float EdgeTessFactor[3]			: SV_TessFactor; // e.g. would be [4] for a quad domain
	float InsideTessFactor : SV_InsideTessFactor; // e.g. would be Inside[2] for a quad domain
};

#define NUM_CONTROL_POINTS 3

[domain("tri")]
DomainOut main(
	HS_CONSTANT_DATA_OUTPUT input,
	float3 domain : SV_DomainLocation,
	const OutputPatch<HullOut, NUM_CONTROL_POINTS> patch)
{
	float2 wind = float2(-1.0, 0.0);
	float A = 0.75;

	DomainOut dout;

	dout.PosF.xy = patch[0].Pos*domain.x + patch[1].Pos*domain.y + patch[2].Pos*domain.z;
	dout.params = patch[0].param*domain.x + patch[1].param*domain.y + patch[2].param*domain.z;
	dout.params.zw = normalize(dout.params.zw);

	float dist = length(camPos - float3(dout.PosF.x, 0.0, dout.PosF.y));

	float3 dP = float3(0.0, 0.0, 0.0);

	dP += wavesDisplacement.SampleLevel(samAnisotropic, float3(dout.PosF.xy / GRID_SIZE.x, 0), 0.0).rbg;
	float3 fft2 = (1.0 - smoothstep(GRID_SIZE.x, 1.2*GRID_SIZE.x, dist))*wavesDisplacement.SampleLevel(samAnisotropic, float3(dout.PosF.xy / GRID_SIZE.y, 1), 0.0).rbg;
	dP += fft2;
	dP *= float3(lambdaV, 1.0, lambdaV);

	float factor = (1.0 - smoothstep(GRID_SIZE.x, 1.2*GRID_SIZE.x, dist))*wavesDisplacement.SampleLevel(samAnisotropic, float3(dout.PosF.xy / GRID_SIZE.y, 1), 0.0).b;
	factor = factor+1.0; // fft
	factor *= factor;

	factor *= (dot(dout.params.zw, wind) + 1.0)*0.5; // wind

	factor *= dout.params.y; // depth

	// add check for [0..1] normalization for A factor
	factor = saturate(factor*A); // A

	float3 gern = 0.0.xxx;
	gern.y = factor*cos(dout.params.x);
	gern.xz -= dout.params.zw*factor*sin(dout.params.x) - 1.5*gern.y*dout.params.zw;
	dout.PosF.z = factor;

	dP = lerp(dP, gern, clamp(2.0*dout.params.y, 0.0, 0.8));

	dout.PosW = float3(dout.PosF.x, 0.0, dout.PosF.y) + dP;

	dout.PosH = mul(float4(dout.PosW, 1.0f), worldToScreenMatrix);

	return dout;
}
