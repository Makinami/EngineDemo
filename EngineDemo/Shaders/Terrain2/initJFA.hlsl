Texture2D<float4> heightmap : register(t0);

RWTexture2D<float4> df : register(u0);

cbuffer JFAstepparams: register(b0)
{
	int step;
	float2 size;
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
	df[DTid.xy] = float4(0.0.xx, heightmap.SampleLevel(trilinear, DTid.xy / size, mip).x > 0.0 ? DTid.xy : 0.0.xx);
}