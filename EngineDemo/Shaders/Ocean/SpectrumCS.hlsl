#include "perFrameCB.hlsli"

RWTexture2DArray<float> Spectrum : register(u0);

cbuffer constBuffer : register(b0)
{
	float4 INVERSE_GRID_SIZE;
	float4 GRID_SIZE;
	float4 MIN_K;
};

static const int gResolution = 256;

static const float PI = 3.14159265359;
static const float G = 9.81;
static const float KM = 370.0;
static const float CM = 0.23;

float square(float x) {
	return x * x;
}

float omega(float k) {
	return sqrt(G * k * (1.0 + square(k / KM)));
}

float tanh(float x) {
	return (1.0 - exp(-2.0 * x)) / (1.0 + exp(-2.0 * x));
}

[numthreads(16, 16, 1)]
void main(int3 DTid : SV_DispatchThreadID)
{
	int2 coordinates = DTid.xy;
	float n = (coordinates.x < gResolution * 0.5) ? coordinates.x : coordinates.x - gResolution;
	float m = (coordinates.y < gResolution * 0.5) ? coordinates.y : coordinates.y - gResolution;
	float2 waveVector = float2(n, m) * INVERSE_GRID_SIZE[DTid.z];

	// cpp
	//float U10 = length(gbWind);
	//float Omega = 0.84;

	//// phase speed
	//float k = length(waveVector);
	//float c = omega(k) / k;

	//// peak
	//float kp = 9.81 * square(Omega / U10);
	//float cp = omega(kp) / kp;

	// friction
	/*float z0 = 3.7e-5 * square(U10) / 9.81 * pow(U10 / cp, 0.9);
	float u_star = 0.41 * U10 / log(10 / z0);

	float alpha_p = 6e-3 * sqrt(Omega);
	float Lpm = exp(-1.25 * square(kp / k));
	float gamma = Omega < 1.0 ? 1.7 : 1.7 + 6.0 * log(Omega);
	float sigma = 0.08 * (1.0 + 4.0 * pow(Omega, -3.0));
	float Gamma = exp(-square(sqrt(k / kp) - 1.0) / (2.0 * square(sigma)));
	float Jp = pow(gamma, Gamma);
	float Fp = Lpm * Jp * exp(-Omega / sqrt(10) * (sqrt(k / kp) - 1.0));
	float Bl = alpha_p * (cp / c) * Fp / 2.0;

	float alpha_m = 0.01 * (u_star < CM ? 1.0 + log(u_star / CM) : 1.0 + 3.0 * log(u_star / CM));
	float Fm = exp(-square(k / KM - 1) / 4.0);
	float Bh = 0.5 * alpha_m * CM / c * Fm * Lpm;

	float a0 = log(2.0) / 4.0;
	float ap = 4.0;
	float am = 0.13 * u_star / CM;
	float delta = tanh(a0 + ap * pow(c / cp, 2.5) + am * pow(CM / c, 2.5));

	float phi = atan2(waveVector.y, waveVector.x);

	float cosPhi = dot(normalize(gbWind), normalize(waveVector));*/

	/*if (waveVector.x < 0.0)
	Bl = Bh = 0.0;
	else
	{
	Bl *= 2.0;
	Bh *= 2.0;
	}*/

	//float S = (Bl + Bh) * (1.0f + delta * (2.0 * cosPhi * cosPhi - 1.0)) / (2.0f * PI * square(square(k)));


	// david
	float k = length(waveVector);

	float U10 = length(gbWind);

	float Omega = 0.84;
	float kp = G * square(Omega / U10);

	float c = omega(k) / k;
	float cp = omega(kp) / kp;

	float Lpm = exp(-1.25 * square(kp / k));
	float gamma = 1.7;
	float sigma = 0.08 * (1.0 + 4.0 * pow(Omega, -3.0));
	float Gamma = exp(-square(sqrt(k / kp) - 1.0) / 2.0 * square(sigma));
	float Jp = pow(gamma, Gamma);
	float Fp = Lpm * Jp * exp(-Omega / sqrt(10.0) * (sqrt(k / kp) - 1.0));
	float alphap = 0.006 * sqrt(Omega);
	float Bl = 0.5 * alphap * cp / c * Fp;

	float z0 = 0.000037 * square(U10) / G * pow(U10 / cp, 0.9);
	float uStar = 0.41 * U10 / log(10.0 / z0);
	float alpham = 0.01 * ((uStar < CM) ? (1.0 + log(uStar / CM)) : (1.0 + 3.0 * log(uStar / CM)));
	float Fm = exp(-0.25 * square(k / KM - 1.0));
	float Bh = 0.5 * alpham * CM / c * Fm * Lpm;

	float a0 = log(2.0) / 4.0;
	float am = 0.13 * uStar / CM;
	float Delta = tanh(a0 + 4.0 * pow(c / cp, 2.5) + am * pow(CM / c, 2.5));

	float cosPhi = dot(normalize(gbWind), normalize(waveVector));

	float S = (1.0 / (2.0 * PI)) * pow(k, -4.0) * (Bl + Bh) * (1.0 + Delta * (2.0 * cosPhi * cosPhi - 1.0));
	//

	float dk = INVERSE_GRID_SIZE[DTid.z];
	float h = sqrt(S / 2.0) * dk;

	if (all(abs(waveVector) < MIN_K[DTid.z])) {
		h = 0.0;
	}

	Spectrum[DTid.xyz] = h;
}