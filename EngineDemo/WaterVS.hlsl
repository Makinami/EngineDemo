cbuffer MatrixBuffer
{
	matrix gWorldProjView;
};

struct VertexIn
{
	float3 PosW : POSITION;
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