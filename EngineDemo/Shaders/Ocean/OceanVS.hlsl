#include "perFrameCB.hlsli" // b1

cbuffer constBuffer : register(b0)
{
	float4 INVERSE_GRID_SIZE;
	float4 GRID_SIZE;
};

Texture2DArray wavesDisplacement : register(t0);
SamplerState samAnisotropic : register(s0);

struct VertexIn
{
	float2 Pos : POSITION;
};

struct VertexOut
{
	float4 PosH : SV_POSITION;
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
	float2 pos = vin.Pos - float2(0.0, screendy);

	float2 u = screenToWorld(pos);
	float2 ux = screenToWorld(pos + float2(gridSize.x, 0.0f));
	float2 uz = screenToWorld(pos + float2(0.0f, gridSize.y));
	float2 dux = abs(ux - u) * 2.0f;
	float2 duz = abs(uz - u) * 2.0f;

	float3 dP = float3(0.0f, 0.0f, 0.0f);
	[unroll(4)]
	for (int i = 0; i < 4; ++i)
		dP += wavesDisplacement.SampleGrad(samAnisotropic, float3(pos / GRID_SIZE[i], i), dux / GRID_SIZE[i], duz / GRID_SIZE[i]).rbg;

	vout.PosH = mul(float4(u.x + dP.x, 0.0 + dP.y, u.y + dP.z, 1.0f), worldToScreenMatrix);
	//vout.PosH = float4(pos, 0.0f, 1.0f);
	return vout;
}