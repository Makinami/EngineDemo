cbuffer cbPerFrameHS
{
	float4 gWorldFrustumPlanes[6];

	float3 gEyePosW;

	// When distance is minimum, the tessellation is maximum.
	// When distance is maximum, the tessellation is minimum.
	float gMinDist;
	float gMaxDist;

	// Exponents for power of 2 tessellation.  The tessellation
	// range is [2^(gMinTess), 2^(gMaxTess)].  Since the maximum
	// tessellation is 64, this means gMaxTess can be at most 6
	// since 2^6 = 64.
	float gMinTess;
	float gMaxTess;

	bool gFrustumCull;
}

// Input control point
struct VSVertexOut
{
	float3 PosW : POSITION;
	float2 Tex : TEXCOORD0;
};

// Output control point
struct HSVertexOut
{
	float3 PosW : POSITION;
	float2 Tex : TEXCOORD0;
};

// Output patch constant data.
struct TessSettings
{
	float EdgeTess[4]			: SV_TessFactor;
	float InsideTess[2]		: SV_InsideTessFactor;
};

#define NUM_CONTROL_POINTS 4

float CalcTessFactor(float3 p)
{
	float d = distance(p, gEyePosW);

	float s = saturate((d - gMinDist) / (gMaxDist - gMinDist));

	return pow(2, (lerp(gMaxTess, gMinTess, s)));
}

// Returns true if the box is completely behind (in negative half space) of plane.
bool AabbBehindPlaneTest(float3 centre, float3 extents, float4 plane)
{
	float3 n = abs(plane.xyz);

	float r = dot(extents, n);

	float s = dot(float4(centre, 1.0f), plane);

	return (s + r) < 0.0f;
}

// Returns true if the box is completely outside the frustum.
bool AabbOutsideFrustumTest(float3 centre, float3 extents, float4 frustumPlanes[6])
{
	[unroll]
	for (int i = 0; i < 6; ++i)
	{
		return AabbBehindPlaneTest(centre, extents, frustumPlanes[i]);
	}

	return false;
}

// Patch Constant Function
TessSettings CalcHSPatchConstants(
	InputPatch<VSVertexOut, NUM_CONTROL_POINTS> patch,
	uint PatchID : SV_PrimitiveID)
{
	TessSettings pt;

	// Frustum test
	float3 vMin = float3(patch[2].PosW.x, patch[2].PosW.y, patch[2].PosW.z);
	float3 vMax = float3(patch[1].PosW.x, patch[1].PosW.y, patch[1].PosW.z);

	float3 boxCentre = 0.5f*(vMin + vMax);
	float3 boxExtents = 0.5f*(vMax - vMin);

	// TEMP: Which is faster?
	/*if (gFrustumCull == 0)
	{
	pt.EdgeTess[0] = 0.0f;
	pt.EdgeTess[1] = 0.0f;
	pt.EdgeTess[2] = 0.0f;
	pt.EdgeTess[3] = 0.0f;

	pt.InsideTess[0] = 0.0f;
	pt.InsideTess[1] = 0.0f;

	return pt;
	}
	else
	{
	if (AabbOutsideFrustumTest(boxCentre, boxExtents, gWorldFrustumPlanes))
	{
	pt.EdgeTess[0] = 2.0f;
	pt.EdgeTess[1] = 2.0f;
	pt.EdgeTess[2] = 2.0f;
	pt.EdgeTess[3] = 2.0f;

	pt.InsideTess[0] = 2.0f;
	pt.InsideTess[1] = 2.0f;

	return pt;
	/*}
	else
	{*/
	float3 e0 = 0.5f*(patch[0].PosW + patch[2].PosW);
	float3 e1 = 0.5f*(patch[0].PosW + patch[1].PosW);
	float3 e2 = 0.5f*(patch[1].PosW + patch[3].PosW);
	float3 e3 = 0.5f*(patch[2].PosW + patch[3].PosW);
	float3 c = 0.25f*(patch[0].PosW + patch[1].PosW + patch[2].PosW + patch[3].PosW);

	pt.EdgeTess[0] = CalcTessFactor(e0);
	pt.EdgeTess[1] = CalcTessFactor(e1);
	pt.EdgeTess[2] = CalcTessFactor(e2);
	pt.EdgeTess[3] = CalcTessFactor(e3);

	pt.InsideTess[0] = CalcTessFactor(c);
	pt.InsideTess[1] = pt.InsideTess[0];

	return pt;
	/*}
	}*/
}

[domain("quad")]
[partitioning("fractional_odd")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(4)]
[patchconstantfunc("CalcHSPatchConstants")]
HSVertexOut main(
	InputPatch<VSVertexOut, NUM_CONTROL_POINTS> ip,
	uint i : SV_OutputControlPointID,
	uint PatchID : SV_PrimitiveID)
{
	HSVertexOut Output;

	// Insert code to compute Output here
	Output.PosW = ip[i].PosW;
	Output.Tex = ip[i].Tex;

	return Output;
}
