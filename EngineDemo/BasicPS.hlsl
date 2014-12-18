struct PixelInputType
{
	float4 Pos : SV_POSITION;
};

float4 main(PixelInputType pin) : SV_TARGET
{
	return float4(0.25f, 0.25f, 0.25f, 1.0f);
}