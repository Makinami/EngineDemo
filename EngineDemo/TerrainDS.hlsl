Texture2D gHeightMap;

cbuffer MatrixBuffer
{
	matrix gViewProj;
};

SamplerState samHeightmap
{
	Filter = MIN_MAG_LINEAR_MIP_POINT;

	AddressU = CLAMP;
	AddressV = CLAMP;
};

struct DomainOut
{
	float4 PosH : SV_POSITION;
	float3 PosW : POSITION;
	float2 Tex : TEXCOORD0;
	float2 TiledTex : TEXCOORD1;
};

// Output control point
struct HullOut
{
	float3 PosW : POSITION;
	float2 Tex : TEXCOORD0;
	float2 BoundsY : TEXCOORD1;
};

// Output patch constant data.
struct TessSettings
{
	float EdgeTessFactor[4]			: SV_TessFactor; // e.g. would be [4] for a quad domain
	float InsideTessFactor[2]			: SV_InsideTessFactor; // e.g. would be Inside[2] for a quad domain
};

#define NUM_CONTROL_POINTS 4

float gTexScale = 50.0f;

[domain("quad")]
DomainOut main(
	TessSettings input,
	float2 domain : SV_DomainLocation,
	const OutputPatch<HullOut, NUM_CONTROL_POINTS> quad)
{
	DomainOut dout;

	dout.PosW = lerp(
		lerp(quad[0].PosW, quad[1].PosW, domain.x),
		lerp(quad[2].PosW, quad[3].PosW, domain.x),
		domain.y);

	dout.Tex = lerp(
		lerp(quad[0].Tex, quad[1].Tex, domain.x),
		lerp(quad[2].Tex, quad[3].Tex, domain.x),
		domain.y);

	dout.TiledTex = dout.Tex*gTexScale;

	dout.PosW.y = gHeightMap.SampleLevel(samHeightmap, dout.Tex, 0).r;

	dout.PosH = mul(float4(dout.PosW, 1.0f), gViewProj);

	return dout;
}
