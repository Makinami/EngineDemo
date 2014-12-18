cbuffer MatrixBuffer
{
	matrix gWorldViewProj;
};

struct VertexInputType
{
	float3 Pos : POSITION;
};

struct PixelInputType
{
	float4 Pos : SV_POSITION;
};

PixelInputType main(VertexInputType vin)
{
	PixelInputType vout;

	vout.Pos = mul(float4(vin.Pos, 1.0f), gWorldViewProj);

	return vout;
}