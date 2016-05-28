cbuffer MatrixBuffer
{
	matrix gViewInverse;
	matrix gProjInverse;
};

struct VertexIn
{
	float3 PosL : POSITION;
};

struct VertexOut
{
	float4 PosH : SV_POSITION;
	float3 Ray : TEXCOORD;
};

VertexOut main(VertexIn vin)
{
	VertexOut vout;

	vout.PosH = float4(vin.PosL, 1.0f);
	vout.Ray = mul(vout.PosH, gProjInverse).xyz;
	vout.Ray = mul(float4(vout.Ray, 0.0f), gViewInverse).xyz;

	return vout;
}