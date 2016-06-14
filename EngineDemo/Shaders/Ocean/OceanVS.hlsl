#include "perFrameCB.hlsli" // b1

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
};

struct VertexOut
{
	float3 PosF : POSITION;
	float3 Pos : TEXTCOORD0;
	//float4 PosH : SV_POSITION;
	//float2 PosF : TEXTCOORD0;
	//float3 PosW : TEXTCOORD1;
};

float2 screenToWorld(float2 vertex)
{
	float3 camDir = normalize(mul(float4(vertex, 0.0f, 1.0f), screenToCamMatrix).xyz);
	float3 worldDir = (mul(float4(camDir, 0.0f), camToWorldMatrix)).xyz;
	float t = -camPos.y / worldDir.y;
	return camPos.xz + t * worldDir.xz;
}

VertexOut main( VertexIn vin )
{
	VertexOut vout;
	
	////float4 pos = mul(float4(vin.Pos.x, 0.0, vin.Pos.y, 1.0), vin.WorldMat);
	//float3 pos = float3(vin.Pos.x, 0.0, vin.Pos.y)*scale;
	//vout.PosF = pos.xz;

	//float3 dP = float3(0.0f, 0.0f, 0.0f);
	//[unroll(4)]
	//for (int i = 0; i < 4; ++i)
	//	dP += wavesDisplacement.SampleLevel(samAnisotropic, float3(pos.xz / GRID_SIZE[i], i), 0.0).rbg;

	//vout.PosW = pos.xyz + dP * float3(lambdaV, 1.0, lambdaV);
	//vout.PosH = mul(float4(vout.PosW, 1.0f), worldToScreenMatrix);

	//vout.Pos = vin.Pos;
	vout.Pos = float3(vin.Pos, length(vin.Pos));
	vout.PosF.xy = vin.Pos*scale + camPos.xz;
	vout.PosF.z = length(camPos - float3(vout.PosF.x, 0.0, vout.PosF.y));

	return vout;
}