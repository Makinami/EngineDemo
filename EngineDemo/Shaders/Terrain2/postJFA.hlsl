RWTexture2D<float4> dfRead : register(u0);
RWTexture2D<float4> dfWrite : register(u1);

cbuffer JFAstepparams: register(b0)
{
	int step;
	int readId;
	int writeId;
	float ratio;
}

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	float4 df = dfRead[DTid.xy];
	df.x = length(df.zw - DTid.xy) * ratio;
	df.y = df.x * 0.1;
	df.zw = normalize(df.zw - DTid.xy);
	dfWrite[DTid.xy] = df;
}