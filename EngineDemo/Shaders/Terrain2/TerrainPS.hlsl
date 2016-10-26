struct VertexOut
{
	float4 PosH : SV_POSITION;
	float3 PosF : TEXTCOORD0;
	float3 PosW : TEXTCOORD1;
};

float4 main(VertexOut pin) : SV_TARGET
{
	if (pin.PosW.y < 0.5) return float4(1.0, 0.8, 0.0, 1.0);
	else if (pin.PosW.y < 20) return float4(0.25, 1.0, 0.0, 1.0);
	else if (pin.PosW.y < 60) return float4(0.15, 0.5, 0.0, 1.0);
	else return float4(0.25.xxx, 1.0);
	return float4(1.0f, 1.0f, 1.0f, 1.0f);
}