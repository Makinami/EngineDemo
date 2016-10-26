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
	float4 PosH : SV_POSITION;
	float3 PosF : TEXTCOORD0;
	float3 PosW : TEXTCOORD1;
};

//static float3 g_gridDim = 0.0f.xxx;
static float2 g_quadScale = 0.0f;

// morphs vertex xy from from high to low detailed mesh position
float2 morphVertex(float2 inPos, float2 vertex, float morphLerpK)
{
	float2 fracPart = (frac((inPos.xy + 0.5) * float2(g_gridDim.y, g_gridDim.y)) * float2(g_gridDim.z, g_gridDim.z));
	return vertex.xy - fracPart * morphLerpK;
}

cbuffer MatrixBuffer : register(b4)
{
	matrix gViewProj;
	float3 camPos;
	float pad;
}

Texture2D<float> gHeightMap : register(t10);

SamplerState bilinear
{
	Filter = MIN_MAG_LINEAR_MIP_POINT;
	AddressU = Clamp;
	AddressV = Clamp;
};

VertexOut main( VertexIn vin ) 
{
	VertexOut vout;

	vout.PosF.xy = vin.Pos * vin.size + float2(vin.offsetx, vin.offsety);
	vout.PosW = float3(vout.PosF.x, 0.0, vout.PosF.y);
	vout.PosF.z = length(camPos - vout.PosW);

	float morphK = 1.0f - clamp(LODConsts[vin.LOD].morphConsts.x - vout.PosF.z * LODConsts[vin.LOD].morphConsts.y, 0.0, 1.0);
	g_quadScale = vin.size.xx;

	float2 morphedVertex = morphVertex(vin.Pos, vin.Pos, morphK);
	vout.PosF.xy = morphedVertex * vin.size + float2(vin.offsetx, vin.offsety);
	vout.PosW = float3(vout.PosF.x, 0.0, vout.PosF.y);
	vout.PosF.z = length(camPos - vout.PosW);

	vout.PosW.y = gHeightMap.SampleLevel(bilinear, vout.PosW.xz/4096.0f+0.5f, 0.0);

	vout.PosH = mul(float4(vout.PosW, 1.0f), gViewProj);

	return vout;
}