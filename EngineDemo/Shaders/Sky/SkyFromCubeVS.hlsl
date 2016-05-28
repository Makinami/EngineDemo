cbuffer sbPerFrameVS
{
	float4x4 gWorldViewProj;
};

struct VertexIn
{
	float3 PosL : POSITION;
};

struct VertexOut
{
	float4 PosH : SV_POSITION;
	float3 PosL : TEXCOORD;
};

VertexOut main( VertexIn vin )
{
	VertexOut vout;

	// Set z = w so that z/w = 1 (i.e., skydome always on far plane)
	vout.PosH = mul(float4(vin.PosL, 1.0f), gWorldViewProj).xyww;

	// Use local vertex position as cubemap lookup vector
	vout.PosL = vin.PosL;

	return vout;
}