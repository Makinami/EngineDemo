struct VertexIn
{
	float3 PosW : POSITION;
};

struct VertexOut
{
	float4 PosH : SV_POSITION;
	float2 uv : TEXTCOORD0;
};

VertexOut main( VertexIn vin ) 
{
	VertexOut vout;
	vout.PosH = float4(vin.PosW, 1.0f);
	vout.uv = vin.PosW.xy * 1.1f;
	
	return vout;
}