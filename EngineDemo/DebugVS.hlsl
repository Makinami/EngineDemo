cbuffer MatrixBuffer
{
	matrix gViewProj;
};

struct VertexIn
{
	float3 PosL : POSITION;
	float2 Tex : TEXCOORD0;
	float2 BoundsY : TEXCOORD1;
};

struct VertexOut
{
	float4 PosH : SV_POSITION;
	float2 Tex  : TEXCOORD;
};

VertexOut main(VertexIn vin)
{
	VertexOut vout;

	vout.PosH = mul(float4(vin.PosL.xy, 1.0, 1.0f), gViewProj);
	vout.Tex = vin.Tex;

	return vout;
}