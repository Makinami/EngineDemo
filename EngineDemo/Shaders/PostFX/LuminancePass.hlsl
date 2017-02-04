Texture2D<float4> frame;

RWTexture2D<float> luminance;

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	luminance[DTid.xy] = log(0.004 + dot(frame[DTid.xy].rgb, float3(0.27, 0.67, 0.06)));
}