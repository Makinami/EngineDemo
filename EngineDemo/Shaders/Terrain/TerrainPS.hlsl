Texture2D gHeightMap : register(t0);
Texture2D gBlendMap : register(t1);
Texture2DArray gLayerMapArray : register(t2);
Texture2D gShadowMap : register(t3);

struct DirectionalLight
{
	float4 Ambient;
	float4 Diffuse;
	float4 Specular;
	float3 Direction;
	float pad;
};

struct Material
{
	float4 Ambient;
	float4 Diffuse;
	float4 Specular; // w = SpecPower
	float4 Reflect;
};

cbuffer cbPerFramePS
{
	float4x4 gViewProj;

	DirectionalLight gDirLight;
	float3 gEyePosW;

	float gTexelCellSpaceU;
	float gTexelCellSpaceV;
	float gWorldCellSpace;

	float2 padding;
}

SamplerState samHeightmap : register(s0);

SamplerState samLinear : register(s1);

SamplerComparisonState samShadow : register(s2);

struct DomainOut
{
	float4 PosH : SV_POSITION;
	float3 PosW : POSITION0;
	float2 Tex : TEXCOORD0;
	float2 TiledTex : TEXCOORD1;
	float4 ShadowPosH : POSITION1;
};

float4 main(DomainOut pin) : SV_TARGET
{
	// Estimate normal and tangent using central differences.
	float2 leftTex = pin.Tex + float2(-gTexelCellSpaceU, 0.0f);
	float2 rightTex = pin.Tex + float2(gTexelCellSpaceU, 0.0f);
	float2 bottomTex = pin.Tex + float2(0.0f, gTexelCellSpaceV);
	float2 topTex = pin.Tex + float2(0.0f, -gTexelCellSpaceV);

	float leftY = gHeightMap.SampleLevel(samHeightmap, leftTex, 0).r;
	float rightY = gHeightMap.SampleLevel(samHeightmap, rightTex, 0).r;
	float bottomY = gHeightMap.SampleLevel(samHeightmap, bottomTex, 0).r;
	float topY = gHeightMap.SampleLevel(samHeightmap, topTex, 0).r;

	float3 tangent = normalize(float3(2.0f*gWorldCellSpace, rightY - leftY, 0.0f));
	float3 bitan = normalize(float3(0.0f, bottomY - topY, 2.0f*gWorldCellSpace));
	float3 normalW = -cross(tangent, bitan);

	// The toEye vector is used in lightning.
	float3 toEye = gEyePosW - pin.PosW;

	// Cache the distance to the eye from this surface point.
	float disToEye = length(toEye);

	// Normalize.
	toEye /= disToEye;

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

	/* Lightning */
	float4 litColour = texColor;

	float4 ambient = float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 diffuse = float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 spec = float4(0.0f, 0.0f, 0.0f, 0.0f);

	float shadow = 0.0f;

	pin.ShadowPosH.xyz /= pin.ShadowPosH.w;

	float depth = pin.ShadowPosH.z;

	const float dx = 1.0f / 2048.0f;

	float percentLit = 0.0f;
	const float2 offset[9] =
	{
		float2(-dx, -dx), float2(0.0f, -dx), float2(dx, -dx),
		float2(-dx, 0.0f), float2(0.0f, 0.0f), float2(dx, 0.0f),
		float2(-dx, dx), float2(0.0f, dx), float2(dx, dx)
	};

	[unroll]
	for (int i = 0; i < 9; ++i)
	{
		//c0 = gShadowMap.Sample(samLinear, pin.ShadowPosH.xy);
		//if (c0.r <= depth) percentLit += 1.0f;
		percentLit += gShadowMap.SampleCmpLevelZero(samShadow, pin.ShadowPosH.xy + offset[i], depth).r;
	}

	//shadow = gShadowMap.SampleCmpLevelZero(samShadow, pin.ShadowPosH.xy, depth).r; //c0 = gShadowMap.Sample(samLinear, pin.ShadowPosH.xy);
	//if (depth <= c0.r) shadow = 1.0f;
	shadow = percentLit / 9.0f;

	float3 lightVec = -gDirLight.Direction;

	Material mat;
	mat.Ambient = float4(1.0f, 1.0f, 1.0f, 1.0f);
	mat.Diffuse = float4(1.0f, 1.0f, 1.0f, 1.0f);
	mat.Specular = float4(0.0f, 0.0f, 0.0f, 64.0f);
	mat.Reflect = float4(0.0f, 0.0f, 0.0f, 0.0f);

	ambient = mat.Ambient * gDirLight.Ambient;

	float diffuseFactor = dot(lightVec, normalW);

	[flatten]
	if (diffuseFactor > 0.0f)
	{
		float3 v = reflect(-lightVec, normalW);
		float specFactor = pow(max(dot(v, toEye), 0.0f), mat.Specular.w);

		diffuse = diffuseFactor * mat.Diffuse * gDirLight.Diffuse;
		spec = specFactor*mat.Specular * gDirLight.Specular;
	}

	diffuse = diffuse*shadow;
	spec = spec*shadow;

	litColour = texColor*(ambient + diffuse) + spec;

	return litColour;
}