RWTexture2D<float4> dfRead : register(u0);
RWTexture2D<float4> dfWrite : register(u1);

cbuffer JFAstepparams: register(b0)
{
	int step;
	int readId;
	int writeId;
	int pad;
}

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	int2 center = DTid.xy;
	float best_dist = 99999.0;
	int2 best_coord = 0.0.xx;

	//[unroll(3)]
	for (int i = -1; i <= 1; ++i)
	{
		//[unroll(3)]
		for (int j = -1; j <= 1; ++j)
		{
			int2 pos = center + int2(i, j)*step;
			int2 ntc = dfRead[pos].zw;
			float d = length(ntc - center);
			if (all(ntc != 0) && (d < best_dist))
			{
				best_dist = d;
				best_coord = ntc;
			}
		}
	}

	dfWrite[center] = float4(length(center-best_coord), 0.0, best_coord);
}