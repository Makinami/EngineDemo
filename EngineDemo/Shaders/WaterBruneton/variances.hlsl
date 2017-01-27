//cbuffer variancesBuffer : register(b0)
//{
//	float4 GRID_SIZE;
//	float slopeVarianceDelta;
//	float3 pad;
//}

cbuffer constBuffer : register(b0)
{
	float4 INVERSE_GRID_SIZE;
	float4 GRID_SIZE;
};

Texture2DArray<float2> spectrum : register(t0);
RWTexture3D<float2> slopeVariances : register(u0);
SamplerState samVarLinear : register(s0);

RWTexture2DArray<float> spectrumTex : register(u8);
Texture2DArray<float> phaseTex : register(t8);

#define FFT_SIZE 256
#define N_SLOPE_VARIANCE 16
#define XM_PI 3.14159265

static const float SCALE = 10.0;

float2 getSlopeVariances(float2 k, float A, float B, float C, float2 spectrumSample)
{
	float w = 1.0 - exp(A * k.x * k.x + B * k.x * k.y + C * k.y * k.y);
	float2 kw = k * w;
	return kw * kw * dot(spectrumSample, spectrumSample) * 2.0;
}

float2 sampleSpectrum(uint3 coordinates)
{
	float phase = phaseTex[coordinates];
	float2 phaseVector;
	sincos(phase, phaseVector.x, phaseVector.y);

	float spec = spectrumTex[coordinates].r;

	return spec*phaseVector;
}

groupshared float2 SpectrumTemp[256][4];

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint index : SV_GroupIndex)
{
	float A = pow(float(DTid.x) / float(N_SLOPE_VARIANCE - 1.0), 4.0) * float(SCALE);
	float C = pow(float(DTid.z) / float(N_SLOPE_VARIANCE - 1.0), 4.0) * float(SCALE);
	float B = (2.0 * float(DTid.y) / float(N_SLOPE_VARIANCE - 1.0) - 1.0) * sqrt(A * C);
	A = -0.5 * A;
	B = -B;
	C = -0.5 * C;

	float2 slope = 0.0.xx;//float2(slopeVarianceDelta, slopeVarianceDelta);
	for (int y = 0; y < FFT_SIZE; ++y)
	{
		[unroll(4)]
		for (int i = 0; i < 4; ++i)
		{
			SpectrumTemp[index][i] = sampleSpectrum(uint3(index, y, i));
		}
		GroupMemoryBarrierWithGroupSync();

		for (int x = 0; x < FFT_SIZE; ++x)
		{
			int i = x >= FFT_SIZE / 2 ? x - FFT_SIZE : x;
			int j = y >= FFT_SIZE / 2 ? y - FFT_SIZE : y;
			float2 k = 2.0 * XM_PI * float2(i, j);

			[unroll(4)]
			for (int i = 0; i < 4; ++i)
				slope += getSlopeVariances(k / GRID_SIZE[i], A, B, C, SpectrumTemp[x][i]);
		}
	}

	slopeVariances[DTid] = slope.xy;
}