cbuffer variancesBuffer : register(c0)
{
	float4 GRID_SIZE;
	float slopeVarianceDelta;
	float3 pad;
}

Texture2DArray<float2> spectrum : register(t0);
RWTexture3D<float4> slopeVariances : register(u0);
SamplerState samVarLinear : register(s0);

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

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	float A = pow(DTid.x / (N_SLOPE_VARIANCE - 1.0), 4.0) * SCALE;
	float C = pow(DTid.z / (N_SLOPE_VARIANCE - 1.0), 4.0) * SCALE;
	float B = (2.0 * DTid.y / (N_SLOPE_VARIANCE - 1.0) - 1.0) * sqrt(A * C);
	A = -0.5 * A;
	B = -B;
	C = -0.5 * C;

	float2 slope = float2(slopeVarianceDelta, slopeVarianceDelta);
	for (int y = 0; y < FFT_SIZE; ++y)
	{
		for (int x = 0; x < FFT_SIZE; ++x)
		{
			int i = x >= FFT_SIZE / 2 ? x - FFT_SIZE : x;
			int j = y >= FFT_SIZE / 2 ? y - FFT_SIZE : y;
			float2 k = 2.0 * XM_PI * float2(i, j);

			slope += getSlopeVariances(k / GRID_SIZE.x, A, B, C, spectrum.SampleLevel(samVarLinear, float3(x, y, 0.0), 0.0));
			slope += getSlopeVariances(k / GRID_SIZE.y, A, B, C, spectrum.SampleLevel(samVarLinear, float3(x, y, 1.0), 0.0));
			slope += getSlopeVariances(k / GRID_SIZE.z, A, B, C, spectrum.SampleLevel(samVarLinear, float3(x, y, 2.0), 0.0));
			slope += getSlopeVariances(k / GRID_SIZE.w, A, B, C, spectrum.SampleLevel(samVarLinear, float3(x, y, 3.0), 0.0));
		}
	}

	slopeVariances[DTid] = float4(slope, C, 1.0);
}