Texture2D gTexture : register(t0);

SamplerState samLinear
{
	Filter = MIN_MAG_MIP_LINEAR;
	AddressU = Wrap;
	AddressV = Wrap;
};

struct VertexOut
{
	float4 PosH : SV_POSITION;
	float2 Tex  : TEXCOORD;
};

float4 main(VertexOut pin) : SV_Target
{
	float4 c = gTexture.Sample(samLinear, pin.Tex).r;
	
	// draw as grayscale
	return float4(c.rrr, 1);
}