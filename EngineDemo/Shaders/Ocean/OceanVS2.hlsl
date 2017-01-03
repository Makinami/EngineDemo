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
	float4 param : TEXTCOORD;
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


	float2 pos = vout.Pos.xy;
	float4 DF = distanceField.SampleLevel(samAnisotropic, (pos + 2048.0f) / 4096.0, 4);

	float depth = -DF.y;
	vout.Pos.z = -depth;
	float pi = 3.141529;
	float g = 9.81;
	float2 wind = float2(-1.0, 0.0);

	float2 gradient = any(DF.zw) ? normalize(normalize(DF.zw) + 0.5*wind) : 0.0.xx;// normalize(float2(10.0, 10.0));

	float wind_dependent = (dot(gradient, wind) + 1.0)*0.5;

	float A = 0.75;
	float lambda = 10.0;// 2.0*pi;// A*2.0*pi; // minimal wavelength
	float w = sqrt(g * 2 * pi / lambda);

	float depth_dependent = 1.0 - saturate(2.0*depth / lambda);

	vout.param.x = 2 * pi*dot(gradient, pos) / lambda - w*time;
	vout.param.y = depth_dependent;
	vout.param.zw = gradient;

	return vout;
}