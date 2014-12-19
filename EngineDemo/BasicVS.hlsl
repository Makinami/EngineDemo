cbuffer MatrixBuffer
{
	matrix gView;
	matrix gProj;
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

	vout.PosH = mul(float4(vin.Pos, 1.0f), gView);
	vout.PosH = mul(vout.PosH, gProj);
	vout.Color = vin.Color;

	return vout;
}