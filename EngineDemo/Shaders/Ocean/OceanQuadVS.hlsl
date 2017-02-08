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

VertexOut main(VertexIn vin)
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


	float3 dP = float3(0.0, 0.0, 0.0);

	dP += wavesDisplacement.SampleLevel(samAnisotropic, float3(vout.PosF.xy / GRID_SIZE.x, 0), 0.0).rbg;
	dP += (1.0 - smoothstep(GRID_SIZE.x, 1.2*GRID_SIZE.x, vout.PosF.z))*wavesDisplacement.SampleLevel(samAnisotropic, float3(vout.PosF.xy / GRID_SIZE.y, 1), 0.0).rbg;
	dP *= float3(lambdaV, 1.0, lambdaV);
	//vout.PosW += dP * float3(lambdaV, 1.0, lambdaV);

	float2 pos = vout.PosW.xz;
	float4 DF = distanceField.SampleLevel(samAnisotropic, (pos + 2048.0f) / 4096.0, 4);

	float depth = DF.y;

	float pi = 3.141529;
	float g = 9.81;
	float2 wind = float2(-1.0, 0.0);

	float2 gradient = any(DF.zw) ? normalize(normalize(DF.zw) + 0.2*wind) : 0.0.xx;// normalize(float2(10.0, 10.0));

	float wind_dependent = (dot(gradient, wind) + 1.0)*0.5;

	float A = 0.75;
	float lambda = 10.0;// 2.0*pi;// A*2.0*pi; // minimal wavelength
	float w = sqrt(g * 2 * pi / lambda);

	float depth_dependent = 1.0 - saturate(2.0*depth / lambda);

	float fft_dependent_noise = (1.0 - smoothstep(GRID_SIZE.x, 1.2*GRID_SIZE.x, vout.PosF.z))*wavesDisplacement.SampleLevel(samAnisotropic, float3(vout.PosF.xy / GRID_SIZE.y, 1), 0.0).b;
	fft_dependent_noise = max((fft_dependent_noise + 0.5), 0.0);

	float k = 2.0*pi / lambda;

	float3 gern = 0.0.xxx;
	gern.y = A*wind_dependent*fft_dependent_noise*depth_dependent*cos(2 * pi*dot(gradient, pos) / lambda - w*time);
	gern.xz -= gradient*A*wind_dependent*fft_dependent_noise*depth_dependent*sin(2 * pi*dot(gradient, pos) / lambda - w*time) - gern.y*gradient;

	float x = frac((2 * pi*dot(gradient, pos) / lambda - w*time) / (2 * pi));
	float y1 = 3.0*x - 2;
	float y2 = -8 * x + 1;
	float foam = saturate(max(y1, y2));

	//vout.PosW.y = foam;

	dP = lerp(dP, gern, saturate(2.0*depth_dependent));

	vout.PosW += dP;

	vout.PosH = mul(float4(vout.PosW, 1.0f), worldToScreenMatrix);

	return vout;
}