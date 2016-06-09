#include "perFrameCB.hlsli" // b1

cbuffer constBuffer : register(b0)
{
	float4 INVERSE_GRID_SIZE;
	float4 GRID_SIZE;
};

Texture2DArray<float4> fftWaves : register(t0);
Texture2DArray<float4> turbulenceSRV : register(t1);
RWTexture2DArray<float4> turbulenceUAV : register(u0);
SamplerState samAnisotropic : register(s0);

#define FFT_SIZE 256
static const float M = 0.4;
static const float k = 4;
static const float fftSize = 256.0;

[numthreads(16, 16, 1)]
void main( int3 DTid : SV_DispatchThreadID )
{
	float4 turbulence;
	int4 d_position = int4(DTid.xy - int2(1, 1), DTid.xy + int2(1, 1));
	
	d_position = modf(d_position / fftSize + 1.0, turbulence) * fftSize;
	turbulence = turbulenceSRV[DTid];

	float Jacobian;

	float2 dDdx = 0.5 * fftSize / GRID_SIZE[DTid.z] * lambda * (fftWaves[uint3(d_position.z, DTid.yz)] - fftWaves[uint3(d_position.x, DTid.yz)]);
	float2 dDdy = 0.5 * fftSize / GRID_SIZE[DTid.z]  * lambda * (fftWaves[uint3(DTid.x, d_position.w, DTid.z)] - fftWaves[uint3(DTid.x, d_position.y, DTid.z)]);

	Jacobian = (1.0 + dDdx.x) * (1.0 + dDdy.y) - dDdx.y * dDdy.x;
	float satJacobian = saturate(k*(-Jacobian + M));

	turbulenceUAV[DTid] = float4(0.0, satJacobian, turbulence.z*exp(-dt) + satJacobian*dt, 0.0);
}