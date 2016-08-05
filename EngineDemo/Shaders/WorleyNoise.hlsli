// modified https://www.shadertoy.com/view/4l2GzW

#ifndef TEXT_SIZE
#define TEXT_SIZE 128
#endif

float r(float n)
{
	return frac(cos(n*89.42)*343.42);
}

float3 r(float3 n)
{
	return float3(r(n.x*23.62 - 300.0 + n.y*34.35 + n.z*26.74), r(n.x*45.13 + 256.0 + n.y*38.89 + n.z*39.45), r(n.x*63.51 + 132.0 + n.y*41.87 + n.z*53.61));
}

float worley(float3 n, float freq, float loop)
{
	float dis = 2.0;
	for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int z = -1; z <= 1; z++)
			{
				float3 p = fmod(floor(n / freq) + float3(x, y, z) + loop, loop);

				float d = length(r(p) + float3(x, y, z) - frac(n / freq));
				if (dis>d)
				{
					dis = d;
				}
			}
		}
	}

	return dis;
}

float fBmWorley(float3 p, float freq, float persistance = 0.5, uint octaves = 0)
{
	if (octaves == 0) octaves = log2(freq);
	float f = 0.0;
	float amp = 1.0;
	float n = 0.0;
	[unroll(8)]
	for (; octaves; --octaves)
	{
		n += amp*worley(p + 131.31*octaves, freq, TEXT_SIZE / freq);
		f += amp;
		amp *= persistance;
		freq *= 0.5;
	}
	return n / f;
}