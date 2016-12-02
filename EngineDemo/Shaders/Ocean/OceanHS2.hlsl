#include "perFrameCB.hlsli" // b1

// Input control point
struct VertexOut
{
	float2 Pos : POSITION;
	float4 param : TEXTCOORD;
};

// Output control point
struct HullOut
{
	float2 Pos : POSITION;
	float4 param : TEXTCOORD;
};

// Output patch constant data.
struct HS_CONSTANT_DATA_OUTPUT
{
	float EdgeTessFactor[3]			: SV_TessFactor; // e.g. would be [4] for a quad domain
	float InsideTessFactor : SV_InsideTessFactor; // e.g. would be Inside[2] for a quad domain
};

#define NUM_CONTROL_POINTS 3
static const float basicTesFactor = 15.0;
static const float declineFactor = 3.0;

// Patch Constant Function
HS_CONSTANT_DATA_OUTPUT CalcHSPatchConstants(
	InputPatch<VertexOut, NUM_CONTROL_POINTS> ip,
	uint PatchID : SV_PrimitiveID)
{
	HS_CONSTANT_DATA_OUTPUT Output;

	if (any(ip[0].param.zw))
	{
		Output.EdgeTessFactor[0] = (ip[1].param.y + ip[2].param.y)*1.5f + 1.0f;
		Output.EdgeTessFactor[1] = (ip[0].param.y + ip[2].param.y)*1.5F + 1.0f;
		Output.EdgeTessFactor[2] = (ip[0].param.y + ip[1].param.y)*1.5f + 1.0f;
		Output.InsideTessFactor = (Output.EdgeTessFactor[0] + Output.EdgeTessFactor[1] + Output.EdgeTessFactor[2]) / 3.0f;
	}
	else
	{
		Output.EdgeTessFactor[0] =
			Output.EdgeTessFactor[1] =
			Output.EdgeTessFactor[2] =
			Output.InsideTessFactor = 0;
	}
	
	return Output;
}

[domain("tri")]
[partitioning("fractional_odd")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("CalcHSPatchConstants")]
HullOut main(
	InputPatch<VertexOut, NUM_CONTROL_POINTS> ip,
	uint i : SV_OutputControlPointID,
	uint PatchID : SV_PrimitiveID)
{
	HullOut Output;

	Output.Pos = ip[i].Pos;
	Output.param = ip[i].param;

	return Output;
}
