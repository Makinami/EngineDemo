cbuffer MatrixBuffer
{
	matrix gViewProj;
};

struct VertexInputType
{
	float4 Pos : POSITION;
};

struct PixelInputType
{
	float4 PosH : SV_POSITION;
	float4 Color : COLOR;
};

PixelInputType main(VertexInputType vin)
{
	PixelInputType vout;

	vout.PosH = mul(float4(vin.Pos.xyz, 1.0f), gViewProj);
	vout.Color = float4(vin.Pos.www, 1.0);

	return vout;
}