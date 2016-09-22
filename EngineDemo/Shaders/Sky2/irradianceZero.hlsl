RWTexture2D<float4> irradiance : register(u0);

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	irradiance[DTid.xy] = 0.0.xxxx;
}