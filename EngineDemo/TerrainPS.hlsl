Texture2D gHeightMap : register(t0);
Texture2D gBlendMap : register(t1);
Texture2DArray gLayerMapArray : register(t2);
/*
struct DirectionalLight
{
	float4 Ambient;
	float4 Diffuse;
	float4 Specular;
	float3 Direction;
	float pad;
};*/
/*
cbuffer cbPerFramePS
{
	float4x4 gViewProj;

	//DirectionalLight gDirLights[3];
	float3 gEyePosW;

	float gTexelCellSpaceU;
	float gTexelCellSpaceV;
	float gWorldCellSpace;

	float2 padding;
}*/

SamplerState samHeightmap : register(s0);

SamplerState samLinear : register(s1);

struct DomainOut
{
	float4 PosH : SV_POSITION;
	float3 PosW : POSITION;
	float2 Tex : TEXCOORD0;
	float2 TiledTex : TEXCOORD1;
};

float4 main(DomainOut pin) : SV_TARGET
{
	// Estimate normal and tangent using central differences.
	/*float2 leftTex = pin.Tex + float2(-gTexelCellSpaceU, 0.0f);
	float2 rightTex = pin.Tex + float2(gTexelCellSpaceU, 0.0f);
	float2 bottomTex = pin.Tex + float2(0.0f, gTexelCellSpaceV);
	float2 topTex = pin.Tex + float2(0.0f, -gTexelCellSpaceV);

	float leftY = gHeightMap.SampleLevel(samHeightmap, leftTex, 0).r;
	float rightY = gHeightMap.SampleLevel(samHeightmap, rightTex, 0).r;
	float bottomY = gHeightMap.SampleLevel(samHeightmap, bottomTex, 0).r;
	float topY = gHeightMap.SampleLevel(samHeightmap, topTex, 0).r;

	float3 tangent = normalize(float3(2.0f*gWorldCellSpace, rightY - leftY, 0.0f));
	float3 bitan = normalize(float3(0.0f, bottomY - topY, 2.0f*gWorldCellSpace));
	float3 normalW = cross(tangent, bitan);

	// The toEye vector is used in lightning.
	float3 toEye = gEyePosW - pin.PosW;

	// Cache the distance to the eye from this surface point.
	float disToEye = length(toEye);

	// Normalize.
	toEye /= disToEye;*/

	/* Texturing */

	//Sample layers in texture array.
	float4 c0 = gLayerMapArray.Sample(samLinear, float3(pin.TiledTex, 0.0f));
	float4 c1 = gLayerMapArray.Sample(samLinear, float3(pin.TiledTex, 1.0f));
	float4 c2 = gLayerMapArray.Sample(samLinear, float3(pin.TiledTex, 2.0f));
	float4 c3 = gLayerMapArray.Sample(samLinear, float3(pin.TiledTex, 3.0f));
	float4 c4 = gLayerMapArray.Sample(samLinear, float3(pin.TiledTex, 4.0f));

	// Sample the blend map.
	float4 t = gBlendMap.Sample(samLinear, pin.Tex);

	// Blend the layers on top of each other.
	float4 texColor = c0;
	texColor = lerp(texColor, c1, t.r);
	texColor = lerp(texColor, c2, t.g);
	texColor = lerp(texColor, c3, t.b);
	texColor = lerp(texColor, c4, t.a);

	return texColor;
}