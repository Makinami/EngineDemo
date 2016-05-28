cbuffer initFFTBuffer
{
	float4 INVERSE_GRID_SIZE;
	float time;
	float3 pad;
};

Texture2DArray<float2> spectrum : register(t0);
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
void main( uint3 DTid : SV_DispatchThreadID )
{
	int x = DTid.x >= FFT_SIZE / 2 ? DTid.x - FFT_SIZE : DTid.x;
	int y = DTid.y >= FFT_SIZE / 2 ? DTid.y - FFT_SIZE : DTid.y;

	float4 spec1 = float4(spectrum[uint3(DTid.xy, 0)], spectrum[uint3(FFT_SIZE - DTid.xy, 0)]);
	float4 spec2 = float4(spectrum[uint3(DTid.xy, 1)], spectrum[uint3(FFT_SIZE - DTid.xy, 1)]);
	float4 spec3 = float4(spectrum[uint3(DTid.xy, 2)], spectrum[uint3(FFT_SIZE - DTid.xy, 2)]);
	float4 spec4 = float4(spectrum[uint3(DTid.xy, 3)], spectrum[uint3(FFT_SIZE - DTid.xy, 3)]);

	float2 k1 = float2(x, y) * INVERSE_GRID_SIZE.x;
	float2 k2 = float2(x, y) * INVERSE_GRID_SIZE.y;
	float2 k3 = float2(x, y) * INVERSE_GRID_SIZE.z;
	float2 k4 = float2(x, y) * INVERSE_GRID_SIZE.w;

	float K1 = length(k1);
	float K2 = length(k2);
	float K3 = length(k3);
	float K4 = length(k4);

	float IK1 = K1 == 0.0 ? 0.0 : 1.0 / K1;
	float IK2 = K2 == 0.0 ? 0.0 : 1.0 / K2;
	float IK3 = K3 == 0.0 ? 0.0 : 1.0 / K3;
	float IK4 = K4 == 0.0 ? 0.0 : 1.0 / K4;

	float2 h1 = getSpectrum(K1, spec1.xy, spec1.zw);
	float2 h2 = getSpectrum(K2, spec2.xy, spec2.zw);
	float2 h3 = getSpectrum(K3, spec3.xy, spec3.zw);
	float2 h4 = getSpectrum(K4, spec4.xy, spec4.zw);

	float4 slope = float4(i(k1.x * h1) - k1.y * h1, i(k2.x * h2) - k2.y * h2);
	fftWaves[uint3(DTid.xy, 0)] = float4(slope.xy * IK1, h1.x - h2.y, 0.0); // grid size 1 displacement
	fftWaves[uint3(DTid.xy, 1)] = float4(slope.zw * IK2, h1.y + h2.x, 0.0); // grid size 2 displacement
	fftWaves[uint3(DTid.xy, 4)] = slope; // grid size 1 & 2 slope variance

	slope = float4(i(k3.x * h3) - k3.y * h3, i(k4.x * h4) - k4.y * h4);
	fftWaves[uint3(DTid.xy, 2)] = float4(slope.xy * IK3, h3.x - h4.y, 0.0); // grid size 3 displacement
	fftWaves[uint3(DTid.xy, 3)] = float4(slope.zw * IK4, h3.y + h4.x, 0.0); // grid size 4 displacement
	fftWaves[uint3(DTid.xy, 5)] = slope; // grid size 3 & 4 slope variance
}