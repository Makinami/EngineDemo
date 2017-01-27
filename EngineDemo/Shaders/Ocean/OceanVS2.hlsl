#include "perFrameCB.hlsli"

struct LODConstsStruct
{
	float size;
	float2 morphConsts;
	float distance;
};

#define MAX_LOD_LEVELS 15

cbuffer LODconstBuffer : register(b3)
{
	float3 g_gridDim;
	float pad1;
	LODConstsStruct LODConsts[MAX_LOD_LEVELS];
}

cbuffer constBuffer : register(b0)
{
	float4 INVERSE_GRID_SIZE;
	float4 GRID_SIZE;
};

Texture2DArray wavesDisplacement : register(t0);
SamplerState samAnisotropic : register(s1);

Texture2D<float4> distanceField : register(t40);
Texture2D<float> heightMap : register(t41);
Texture2D<float4> genHM : register(t42);

struct VertexIn
{
	float2 Pos : POSITION;
	float offsetx : PSIZE0;
	float offsety : PSIZE1;
	uint LOD : BLENDINDICES;
	float size : PSIZE2;
};

struct VertexOut
{
	float3 Pos : POSITION;
	float4 param : TEXTCOORD; //  x- argument sincon; y - zaleznosc od glebokosci
};

//static float3 g_gridDim = 0.0f.xxx;
static float2 g_quadScale = 0.0f;

// morphs vertex xy from from high to low detailed mesh position
float2 morphVertex(float2 inPos, float2 vertex, float morphLerpK)
{
	float2 fracPart = (frac((inPos.xy + 0.5) * float2(g_gridDim.y, g_gridDim.y)) * float2(g_gridDim.z, g_gridDim.z));
	return vertex.xy - fracPart * morphLerpK;
}

VertexOut main(VertexIn vin)
{
	VertexOut vout;
	vout.Pos.xy = vin.Pos * vin.size + float2(vin.offsetx, vin.offsety);
	float3 PosW = float3(vout.Pos.x, 0.0, vout.Pos.y);
	float dist = length(camPos - PosW);

	float morphK = 1.0f - clamp(LODConsts[vin.LOD].morphConsts.x - dist * LODConsts[vin.LOD].morphConsts.y, 0.0, 1.0);
	g_quadScale = vin.size.xx;

	float2 morphedVertex = morphVertex(vin.Pos, vin.Pos, morphK);
	vout.Pos.xy = morphedVertex * vin.size + float2(vin.offsetx, vin.offsety);
	PosW = float3(vout.Pos.x, 0.0, vout.Pos.y);
	//vout.Pos.z = length(camPos - PosW);


	float mip = mipmap;
	float2 pos = vout.Pos.xy;
	float4 DF = distanceField.SampleLevel(samAnisotropic, (pos + 2048.0f) / 4096.0, mip);
	//DF = genHM.SampleLevel(samAnisotropic, (pos + 2048.0f) / 4096.0f, mip).yxzw;

	float4 height;
	height.x = heightMap.SampleLevel(samAnisotropic, (pos + 2048.0f + float2(-0.5, 0.0)) / 4096.0, mip);
	height.y = heightMap.SampleLevel(samAnisotropic, (pos + 2048.0f + float2(0.5, 0.0)) / 4096.0, mip);
	height.z = heightMap.SampleLevel(samAnisotropic, (pos + 2048.0f + float2(0.0, -0.5)) / 4096.0, mip);
	height.w = heightMap.SampleLevel(samAnisotropic, (pos + 2048.0f + float2(0.0, 0.5)) / 4096.0, mip);



	float depth = -DF.y;
	//depth = -heightMap.SampleLevel(samAnisotropic, (pos + 2048.0f) / 4096.0, 0);
	vout.Pos.z = -depth;
	float pi = 3.141529;
	float g = 9.81;
	float2 wind = normalize(gbWind);// float2(1.0, 0.0);

	float2 gradient = any(DF.zw) ? normalize(normalize(DF.zw) + 0.0*wind) : 0.0.xx;// normalize(float2(10.0, 10.0));

	//gradient = -normalize(float2(height.x - height.y, height.z - height.w));

	float wind_dependent = (dot(gradient, wind) + 1.0)*0.5;

	float A = 0.75;
	A = 0.27 * dot(gbWind, gbWind) / 9.81;
	A *= 0.5;
	float lambda = A*2.0*pi; // minimal wavelength
	//lambda = max(A*2.0*pi, pi);
	float w = sqrt(g * 2 * pi / 10.0);

	float depth_dependent = 1.0-saturate(2.0*depth / lambda);

	vout.param.x = -DF.x/(A*argHDependency) - w*time;
	vout.param.y = depth_dependent;
	vout.param.zw = gradient;

	return vout;
}