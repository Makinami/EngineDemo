Texture2DArray gLayerMapArray : register(t2);

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

sampler samLinear : register(s0);

PixelOut main(VertexOut pin)
{
	PixelOut pout;

	if (pin.PosW.y < 0.5) pout.Colour = gLayerMapArray.Sample(samLinear, float3(pin.PosF.xy/30.0, 3));
	else if (pin.PosW.y < 20) pout.Colour = gLayerMapArray.Sample(samLinear, float3(pin.PosF.xy / 30.0, 0));
	else if (pin.PosW.y < 60) pout.Colour = gLayerMapArray.Sample(samLinear, float3(pin.PosF.xy / 30.0, 2));
	else pout.Colour = gLayerMapArray.Sample(samLinear, float3(pin.PosF.xy / 30.0, 2));
	pout.Colour.a = pin.PosW.y/100;
	pout.Normal = pin.Normal.xz;

	return pout;
}