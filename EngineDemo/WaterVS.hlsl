cbuffer MatrixBuffer
{
	matrix gWorldProjView;
};

struct VertexIn
{
	float3 PosW : POSITION0;
};

SamplerState samHeightmap : register(s0);

Texture2D gFFTOutput : register(t0);

struct VertexOut
{
	float4 PosH : SV_POSITION;
};

VertexOut main( VertexIn vin )
{
	VertexOut dout;

	float3 displacement = gFFTOutput[vin.PosW.xz].rgb;
	
	vin.PosW += displacement;

	dout.PosH = mul(float4(vin.PosW, 1.0f), gWorldProjView);

	return dout;
}