cbuffer MatrixBuffer
{
	matrix gWorldProjView;
};

struct VertexIn
{
	float3 PosW : POSITION0;
	float3 Normal : NORMAL;
	float3 hTilde0 : TEXCOORD0;
	float3 hTilde0mkconj : TEXCOORD1;
	float3 Original : POSITION1;
};

struct VertexOut
{
	float4 PosH : SV_POSITION;
};

VertexOut main( VertexIn vin )
{
	VertexOut dout;

	dout.PosH = mul(float4(vin.PosW, 1.0f), gWorldProjView);

	return dout;
}