Texture2D gHeightMap;

cbuffer MatrixBuffer
{
	matrix gViewProj;
	matrix gShadowTrans;
};

SamplerState samHeightmap : register(s0);

struct DomainOut
{
	float4 PosH : SV_POSITION;
	float2 Tex : TEXCOORD0;
};

// Output control point
struct HullOut
{
	float3 PosW : POSITION;
	float2 Tex : TEXCOORD0;
};

// Output patch constant data.
struct TessSettings
{
	float EdgeTessFactor[4]			: SV_TessFactor; // e.g. would be [4] for a quad domain
	float InsideTessFactor[2]			: SV_InsideTessFactor; // e.g. would be Inside[2] for a quad domain
};

#define NUM_CONTROL_POINTS 4

#define gTexScale 50.0f;

[domain("quad")]
DomainOut main(
	TessSettings input,
	float2 domain : SV_DomainLocation,
	const OutputPatch<HullOut, NUM_CONTROL_POINTS> quad)
{
	DomainOut dout;

	float3 temp = lerp(
		lerp(quad[0].PosW, quad[1].PosW, domain.x),
		lerp(quad[2].PosW, quad[3].PosW, domain.x),
		domain.y);

	float2 temp2 = lerp(
		lerp(quad[0].Tex, quad[1].Tex, domain.x),
		lerp(quad[2].Tex, quad[3].Tex, domain.x),
		domain.y);

	temp += gHeightMap.SampleLevel(samHeightmap, temp2, 0).rgb;

	dout.PosH = mul(float4(temp, 1.0f), gViewProj);

	dout.Tex = temp2;

	return dout;
}
