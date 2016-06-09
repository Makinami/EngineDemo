#include "perFrameCB.hlsli" // b1

cbuffer constBuffer : register(b0)
{
	float4 INVERSE_GRID_SIZE;
	float4 GRID_SIZE;
};

Texture2DArray wavesDisplacement : register(t0);
SamplerState samAnisotropic : register(s1);

struct VertexIn
{
	float2 Pos : POSITION;
};

struct VertexOut
{
	float4 PosH : SV_POSITION;
	float2 PosF : TEXTCOORD0;
	float3 PosW : TEXTCOORD1;
};

VertexOut main( VertexIn vin )
{
	VertexOut vout;

	float3 dP = float3(0.0f, 0.0f, 0.0f);
	[unroll(4)]
	for (int i = 0; i < 4; ++i)
		dP += wavesDisplacement.SampleLevel(samAnisotropic, float3(vin.Pos / GRID_SIZE[i], i), 0.0).rbg;

	vout.PosF = vin.Pos;
	vout.PosW = float3(vin.Pos.x, 0.0f, vin.Pos.y) + dP;
	vout.PosH = mul(float4(vout.PosW, 1.0f), worldToScreenMatrix);
	return vout;
}