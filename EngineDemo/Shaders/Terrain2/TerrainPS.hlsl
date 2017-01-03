struct VertexOut
{
	float4 PosH : SV_POSITION;
	float3 PosF : TEXTCOORD0;
	float3 PosW : TEXTCOORD1;
	float3 Normal : NORMAL;
};

struct PixelOut
{
	float4 Colour : SV_TARGET0;
	float2 Normal : SV_TARGET1;
};

PixelOut main(VertexOut pin)
{
	PixelOut pout;

	if (pin.PosW.y < 0.5) pout.Colour = float4(1.0, 0.8, 0.0, 1.0);
	else if (pin.PosW.y < 20) pout.Colour = float4(0.25, 1.0, 0.0, 1.0);
	else if (pin.PosW.y < 60) pout.Colour = float4(0.15, 0.5, 0.0, 1.0);
	else pout.Colour =  float4(0.25.xxx, 1.0);
	pout.Colour.a = pin.PosW.y/100;
	pout.Normal = pin.Normal.xz;

	return pout;
}