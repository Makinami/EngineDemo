cbuffer initFFTBuffer
{
	float4 INVERSE_GRID_SIZE;
	float time;
	float3 pad;
};

Texture2D<float4> spectrum12 : register(t0);
Texture2D<float4> spectrum34 : register(t1);
RWTexture2DArray<float4> fftWaves : register(u0);

#define FFT_SIZE 256

float2 getSpectrum(float k, float2 s0, float2 s0c)
{
	float w = sqrt(9.81 * k * (1.0 + k * k / (370.0 * 370.0)));
	float s, c;
	sincos(w * time, s, c);
	return float2((s0.x + s0c.x) * c - (s0.y + s0c.y) * s, (s0.x - s0c.x) * s + (s0.y - s0c.y) * c);
}

float2 i(float2 z) // i * z (complex)
{
	return float2(-z.y, z.x);
}

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	int x = DTid.x >= FFT_SIZE / 2 ? DTid.x - FFT_SIZE : DTid.x;
	int y = DTid.y >= FFT_SIZE / 2 ? DTid.y - FFT_SIZE : DTid.y;

	float4 s12 = spectrum12[DTid.xy];

	fftWaves[uint3(DTid.xy, 0)] = s12; 
}