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
	LODConstsStruct LODConsts[MAX_LOD_LEVELS];
}

cbuffer constBuffer : register(b0)
{
	float4 INVERSE_GRID_SIZE;
	float4 GRID_SIZE;
};

Texture2DArray wavesDisplacement : register(t0);
SamplerState samAnisotropic : register(s1);

struct VertexIn
{
	float2 Pos : POSITION;
	float offsetx : PSIZE0;
	float offsety : PSIZE1;
	uint LOD : BLENDINDICES;
};

struct VertexOut
{
	float4 PosH : SV_POSITION;
	float3 PosF : TEXTCOORD0;
	float3 PosW : TEXTCOORD1;
};

static float3 g_gridDim = 0.0f.xxx;
static float2 g_quadScale = 0.0f;

// morphs vertex xy from from high to low detailed mesh position
float2 morphVertex(float2 inPos, float2 vertex, float morphLerpK)
{
	float2 fracPart = (frac((inPos.xy+0.5) * float2(g_gridDim.y, g_gridDim.y)) * float2(g_gridDim.z, g_gridDim.z)) * g_quadScale.xy;
	return vertex.xy - fracPart * morphLerpK;
}

VertexOut main( VertexIn vin )
{
	VertexOut vout;
	vout.PosF.xy = vin.Pos * LODConsts[vin.LOD].size + float2(vin.offsetx, vin.offsety);
	vout.PosW = float3(vout.PosF.x, 0.0, vout.PosF.y);
	vout.PosF.z = length(camPos - vout.PosW);

	float morphK = 1.0f - clamp(LODConsts[vin.LOD].morphConsts.x - vout.PosF.z * LODConsts[vin.LOD].morphConsts.y, 0.0, 1.0);
	g_gridDim.x = 4.0;
	g_gridDim.y = g_gridDim.x / 2.0f;
	g_gridDim.z = 1.0f / g_gridDim.y;
	g_quadScale = LODConsts[vin.LOD].size.xx;

	float2 morphedVertex = morphVertex(vin.Pos, vout.PosW.xz, morphK);
	vout.PosF.z = morphK;
	vout.PosW.xz = morphedVertex;

	vout.PosH = mul(float4(vout.PosW, 1.0f), worldToScreenMatrix);

	float3 dP = 0.0.xxx;
	return vout;
}