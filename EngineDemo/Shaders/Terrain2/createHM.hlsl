RWTexture2D<float4> heightMap : register(u0);

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	float2 gradient;
	float dis = length(DTid.xy - float2(2048, 2048));
	dis = dis - 1024.0;
	gradient = normalize(DTid.xy - float2(2048, 2048));
	if (dis > 0)
	{
		gradient *= -1.0;
	}
	dis = abs(dis);
	if (DTid.y > 2048)
	{
		dis = 8192;
		if (dis > length(DTid.xy - float2(1024, 2048)))
		{
			dis = length(DTid.xy - float2(1024, 2048));
			gradient = -normalize(DTid.xy - float2(1024, 2048));
		}
		if (dis > length(DTid.xy - float2(3072, 2048)))
		{
			dis = length(DTid.xy - float2(3072, 2048));
			gradient = -normalize(DTid.xy - float2(3072, 2048));
		}
	}
	dis = (400 - dis, 0.0)/10.0;

	heightMap[DTid.xy] = float4(dis, 0.0, gradient);
}