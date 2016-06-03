cbuffer WaterParams : register(c0)
{
	matrix screenToCamera; // screen space to camera space
	matrix cameraToWorld; // camera space to world space
	matrix worldToScreen; // world space to screen space
	float3 worldCamera; // camera position
	float normals;
	float3 worldSunDir; // sun direction in world space
	float choppy;
	float4 GRID_SIZES;
	float3 seaColour;
	float pad;
	float2 gridSize;
};

Texture2DArray gDisplacement : register(t0);
SamplerState samFFTMap : register(s2);

struct VertexIn
{
	float3 Pos : POSITION;
};

struct VertexOut
{
	float4 PosH : SV_POSITION;
	float3 PosW : TEXTCOORD0;
	float2 u : TEXTCOORD1;
};

float2 oceanPos(float3 vertex)
{
	float3 cameraDir = normalize(mul(float4(vertex, 1.0f), screenToCamera).xyz);
	float3 worldDir = (mul(float4(cameraDir, 0.0f), cameraToWorld)).xyz;
	float t = -worldCamera.y / worldDir.y;
	return worldCamera.xz + t * worldDir.xz;
}

VertexOut main( VertexIn vin )
{
	VertexOut vout;
	float2 u = oceanPos(vin.Pos);
	float2 ux = oceanPos(vin.Pos + float3(gridSize.x, 0.0f, 0.0f));
	float2 uz = oceanPos(vin.Pos + float3(0.0f, gridSize.y, 0.0f));
	float2 dux = abs(ux - u) * 2.0f;
	float2 duz = abs(uz - u) * 2.0f;

	float3 dP = float3(0.0f, 0.0f, 0.0f);
	[unroll(4)]
	for (int i = 0; i < 4; ++i)
		dP += gDisplacement.SampleGrad(samFFTMap, float3(u / GRID_SIZES[i], i), dux / GRID_SIZES[i], duz / GRID_SIZES[i]).rbg;

	if (choppy <= 0.0) dP = float3(0.0f, dP.y, 0.0f);
	//dP = float3(0.0, 0.0, 0.0);
	vout.PosW = float3(u.x, 0.0f, u.y) + dP;
	vout.PosH = mul(float4(vout.PosW, 1.0f), worldToScreen).xyzw;
	vout.u = u;

	return vout;
}