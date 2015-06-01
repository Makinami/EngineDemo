cbuffer MatrixBuffer
{
	matrix gViewProj;
};

struct VertexInputType
{
	float3 Pos : POSITION;
	float4 Color : COLOR;
};

struct PixelInputType
{
	float4 PosH : SV_POSITION;
	float4 Color : COLOR;
};

PixelInputType main(VertexInputType vin)
{
	PixelInputType vout;

	vout.PosH = mul(float4(vin.Pos, 1.0f), gViewProj);
	vout.Color = vin.Color;

	return vout;
}