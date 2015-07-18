Texture2D<float3> FFTSourceR : register(t0);
RWTexture2D<float3> Displacement : register(u0);

[numthreads(256, 1, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	float lambda = -1.0f;
	float sign;
	if ((DTid.x + DTid.y) & 1) sign = -1.0f;
	else sign = 1.0f;

	float3 dis;
	float3 FFT = FFTSourceR[DTid.xy]*sign;

	dis.x = FFT.x*lambda;
	dis.y = FFT.y;
	dis.z = FFT.z*lambda;

	Displacement[DTid.xy] = dis;
}