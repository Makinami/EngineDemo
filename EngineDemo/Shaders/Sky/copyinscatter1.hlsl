Texture3D<float3> deltaSR;
Texture3D<float3> deltaSM;

RWTexture3D<float4> inscatter;

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{	
	inscatter[DTid] = float4(deltaSR[DTid], deltaSM[DTid].r);
}