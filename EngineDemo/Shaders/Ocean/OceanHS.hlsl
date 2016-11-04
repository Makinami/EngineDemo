#include "perFrameCB.hlsli" // b1

// Input control point
struct VertexOut
{
	float3 PosF : POSITION;
	float3 Pos : TEXTCOORD0;
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
static const float basicTesFactor = 15.0;
static const float declineFactor = 3.0;

// Patch Constant Function
HS_CONSTANT_DATA_OUTPUT CalcHSPatchConstants(
	InputPatch<VertexOut, NUM_CONTROL_POINTS> ip,
	uint PatchID : SV_PrimitiveID)
{
	HS_CONSTANT_DATA_OUTPUT Output;

	// NOTE: Maybe some other frustum culling. Or leave as is...
	// 3x += step - very slow

	float variableTesFactor = max(basicTesFactor * (1.0 - smoothstep(0.1, 0.55, (ip[0].Pos.z + ip[2].Pos.z) / 2.0)), 1);;

	// NOTE: Think about other tesselation curve
	// Insert code to compute Output here
	Output.EdgeTessFactor[0] = variableTesFactor;
	Output.EdgeTessFactor[1] = basicTesFactor;
	Output.EdgeTessFactor[2] = basicTesFactor;
	Output.InsideTessFactor = variableTesFactor; // e.g. could calculate dynamic tessellation factors instead

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

	// Insert code to compute Output here
	Output.PosF = ip[i].PosF;

	return Output;
}
