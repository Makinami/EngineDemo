#include "perFrameCB.hlsli" // b1

cbuffer constBuffer : register(b0)
{
	float4 INVERSE_GRID_SIZE;
	float4 GRID_SIZE;
};

static const float PI = 3.141592657f;

Texture2DArray wavesDisplacement : register(t0);
SamplerState samAnisotropic : register(s1);

struct DomainOut
{
	float4 PosH  : SV_POSITION;
	float3 PosF : TEXTCOORD0;
	float3 PosW : TEXTCOORD1;
};

// Output control point
struct HullOut
{
	float3 PosF : POSITION;
};


// Output patch constant data.
struct HS_CONSTANT_DATA_OUTPUT
{
	float EdgeTessFactor[3]			: SV_TessFactor; // e.g. would be [4] for a quad domain
	float InsideTessFactor : SV_InsideTessFactor; // e.g. would be Inside[2] for a quad domain
};

#define NUM_CONTROL_POINTS 3

[domain("tri")]
DomainOut main(
	HS_CONSTANT_DATA_OUTPUT input,
	float3 domain : SV_DomainLocation,
	const OutputPatch<HullOut, NUM_CONTROL_POINTS> patch)
{
	DomainOut dout;

	dout.PosF = float3(
		patch[0].PosF*domain.x + patch[1].PosF*domain.y + patch[2].PosF*domain.z);

	float dist = dout.PosF.z;
	float amp = 1.0;

	float depth = dout.PosF.x / -1;
	float sigwaveH = 0.688;

	float omega = PI / 4.0;
	float2 shoredir = float2(1.0, 0.0);
	float2 winddir = float2(1.0, 0.0); // NOTE: it should have magnitude of 2pi/lambda
	float2 H = normalize(winddir + 2.0*shoredir); // so no normalize, but ... for now...
	//if (depth < -5.0) amp = 0.0;
	float3 dP = float3(0.0, 0.0, 0.0);
	float arg = fmod(dot(H, dout.PosF.xz) + omega*time, 2 * PI);
	dP.y = (depth+1.0)*sigwaveH/1.5 *amp*cos(arg);
	dP.x = -0.3*amp*sin(arg)-dP.y*H.x;

	//depth = (dout.PosF.x + lambdaV*dP.x) / -1;	
	
	//dP.y = max(min(depth + 0.001, sigwaveH*((-0.4*pow(cos(arg) - 1.0, 2.0) + 1.2))), dP.y);

	//dP.y = max(min(depth+0.001, sigwaveH), dP.y);

	//dP.y = min(dP.y, 0.5*sigwaveH);

	dout.PosW = float3(dout.PosF.x, 0.0, dout.PosF.y) + dP * float3(lambdaV, 1.0, lambdaV);
	dout.PosH = mul(float4(dout.PosW, 1.0f), worldToScreenMatrix);

	return dout;
}
