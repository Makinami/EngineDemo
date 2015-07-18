cbuffer MatrixBuffer
{
	matrix gWorld;
};

SamplerState samFFTmap : register(s0);

struct VertexIn
{
	float3 PosW : POSITION0;
	float2 Tex : TEXCOORD0;
};

Texture2D gFFTOutput : register(t0);

struct VertexOut
{
	float3 PosW : POSITION0;
	float2 Tex : TEXCOORD0;
};

VertexOut main( VertexIn vin )
{
	VertexOut dout;

	//float3 displacement = gFFTOutput.SampleLevel(samFFTmap, vin.PosW.xz, 0).rgb;
	
	//vin.PosW.xyz += displacement;

	dout.PosW = mul(float4(vin.PosW, 1.0f), gWorld);
	//dout.PosW = vin.PosW;
	dout.Tex = vin.Tex;

	return dout;
}