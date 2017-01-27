Texture2D<float> heighmap : register(t1);
Texture2D<float4> dfRead : register(t0);
RWTexture2D<float4> dfWrite : register(u0);

cbuffer JFAstepparams: register(b0)
{
	int step;
	int readId;
	int writeId;
	float mip;
}

SamplerState trilinear
{
	Filter = MIN_MAG_MIP_LINEAR;
	AddressU = Clamp;
	AddressV = Clamp;
};

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	float4 df = dfRead[DTid.xy];
	float height = heighmap.SampleLevel(trilinear, (float2(DTid.xy) + 0.5f) / 4096.0f, mip);
	if (height <= 0.2f)
	{
		float dx = heighmap.SampleLevel(trilinear, (float2(DTid.x + 1, DTid.y) + 0.5f) / 4096.0f, mip) - heighmap.SampleLevel(trilinear, (float2(DTid.x - 1, DTid.y) + 0.5f) / 4096.0f, mip);
		float dy = heighmap.SampleLevel(trilinear, (float2(DTid.x, DTid.y + 1) + 0.5f) / 4096.0f, mip) - heighmap.SampleLevel(trilinear, (float2(DTid.x, DTid.y - 1) + 0.5f) / 4096.0f, mip);
		df.zw = (dx || dy) ? normalize(float2(dx, dy)) : 0.0.xx;
		dfWrite[DTid.xy] = float4(df.r, max(-df.g, height), df.ba);
	}
	else
		dfWrite[DTid.xy] = float4(0.0, height, 0.0.xx);
}