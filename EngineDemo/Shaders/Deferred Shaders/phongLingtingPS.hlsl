Texture2D<float4> gbColour : register(t0);
Texture2D<float2> gbNormals : register(t1);
Texture2D<float> gbDepth : register(t2);

SamplerState samBilinear : register(s0);

cbuffer cbPerFramePS : register(b0)
{
	float3 bCameraPos;
	float pad1;
	float3 bSunDir;
	float pad2;
	float4 gProj;
};

struct VertexOut
{
	float4 PosH : SV_POSITION;
	float3 Ray : TEXCOORD;
};

struct PixelOut
{
	float4 Colour : SV_TARGET;
	float Depth : SV_DEPTH;
};

PixelOut main(VertexOut pin) 
{
	float3 V = -normalize(pin.Ray);

	float3 N;
	N.xz = gbNormals[pin.PosH.xy];
	N.y = sqrt(1.0 - N.x*N.x - N.z*N.z);

	float3 R = reflect(-bSunDir, N);

	float3 Colour = gbColour[pin.PosH.xy].rgb;

	PixelOut result;
	result.Colour = float4(Colour*pow(saturate(dot(R, V)), 2.0)*5.0 + Colour*0.5, 1.0f);
	result.Depth = gbDepth[pin.PosH.xy];

	return result;
}