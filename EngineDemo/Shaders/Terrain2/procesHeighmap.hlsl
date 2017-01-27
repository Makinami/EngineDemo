Texture2D<float> heighmapRaw : register(t0);
Texture2D<float4> df : register(t1);

RWTexture2D<float> heighmap : register(u0);

SamplerState trilinear
{
	Filter = MIN_MAG_MIP_LINEAR;
	AddressU = Clamp;
	AddressV = Clamp;
};

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	float hm = heighmapRaw.SampleLevel(trilinear, (DTid.xy + 0.5f) / 4096.0f, 0);

	heighmap[DTid.xy] = hm > 0.0 ? max(sqrt(hm)*128.0,0.0) : -df.SampleLevel(trilinear, (DTid.xy + 0.5f) / 4096.0f, 0).y;
}