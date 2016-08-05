// modified http://http.developer.nvidia.com/GPUGems2/gpugems2_chapter26.html

Texture3D<int3> gradTex : register(s0);

float3 GetGradient(uint3 p, uint3 loop)
{
	p %= 128;
	p %= loop;
	return gradTex[p];
}

float GetContribution(int3 grad, float3 p)
{
	return dot(grad, p);
}

float3 fade(float3 t)
{
	return t*t*t*(t*(t*6.0 - 15.0) + 10.0);
}

float PerlinNoise(float3 p, uint3 loop)
{
	float3 P = floor(p);
	p -= P;
	float3 f = fade(p);

	return lerp(
		lerp(lerp(GetContribution(GetGradient(P + uint3(0, 0, 0), loop), p + float3( 0, 0, 0)),
				  GetContribution(GetGradient(P + uint3(1, 0, 0), loop), p + float3(-1, 0, 0)), f.x),
			 lerp(GetContribution(GetGradient(P + uint3(0, 1, 0), loop), p + float3( 0,-1, 0)),
				  GetContribution(GetGradient(P + uint3(1, 1, 0), loop), p + float3(-1,-1, 0)), f.x), f.y),
		lerp(lerp(GetContribution(GetGradient(P + uint3(0, 0, 1), loop), p + float3( 0, 0,-1)),
				  GetContribution(GetGradient(P + uint3(1, 0, 1), loop), p + float3(-1, 0,-1)), f.x),
			 lerp(GetContribution(GetGradient(P + uint3(0, 1, 1), loop), p + float3( 0,-1,-1)),
				  GetContribution(GetGradient(P + uint3(1, 1, 1), loop), p + float3(-1,-1,-1)), f.x), f.y),
		f.z
	);
}

float fBmPerlin(float3 p, float freq, float persistance = 0.5, uint octaves = 0)
{
	if (octaves == 0) octaves = log2(freq);
	float f = 0.0;
	float amp = 1.0;
	float n = 0.0;
	[unroll(8)]
	for (; octaves; --octaves)
	{
		n += amp*PerlinNoise(p / freq + 131.31*octaves, uint3(128, 128, 128) / uint(freq));
		f += amp;
		amp *= persistance;
		freq *= 0.5;
	}
	return n / f;
}