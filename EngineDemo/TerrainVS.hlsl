Texture2D gHeightMap;

SamplerState samHeightmap : register(s0);

struct VertexIn
{
	float3 PosL : POSITION;
	float2 Tex : TEXCOORD0;
	float2 BoundsY : TEXCOORD1;
};

struct VertexOut
{
	float3 PosW : POSITION;
	float2 Tex : TEXCOORD0;
	float2 BoundsY : TEXCOORD1;
};

VertexOut main( VertexIn vin )
{
	VertexOut vout;

	vout.PosW = vin.PosL;

	vout.PosW.y = gHeightMap.SampleLevel(samHeightmap, vin.Tex, 0).r;
	
	vout.Tex = vin.Tex;
	vout.BoundsY = vin.BoundsY;

	return vout;
}