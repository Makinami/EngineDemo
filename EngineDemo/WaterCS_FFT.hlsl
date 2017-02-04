//--------------------------------------------------------------------------------------
// Copyright 2014 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.
//--------------------------------------------------------------------------------------

// Input Preprocess Defines:
// BUTTERFLY_COUNT: number of passes to perform
// ROWPASS: defined for tranformation along the x axis
// LENGTH: pixel length of row or column

#define LENGTH 256
#define BUTTERFLY_COUNT 8

#define ROWPASS col_row_pass.x == 0
#define COLPASS col_row_pass.x == 1
cbuffer FFTParameters
{
	uint4 col_row_pass; // 0 - row, 1 - col
};

Texture2D<float3> FFTSourceR  : register(t0);
Texture2D<float3> FFTSourceI  : register(t1);
RWTexture2D<float3> FFTTargetR  : register(u0);
RWTexture2D<float3> FFTTargetI  : register(u1);

static const float PI = 3.14159265f;

void GetButterflyValues(uint passIndex, uint x, out uint2 indices, out float2 weights)
{
	int sectionWidth = 2 << passIndex;
	int halfSectionWidth = sectionWidth / 2;

	int sectionStartOffset = x & ~(sectionWidth - 1);
	int halfSectionOffset = x & (halfSectionWidth - 1);
	int sectionOffset = x & (sectionWidth - 1);

	sincos(2.0*PI*sectionOffset / (float)sectionWidth, weights.y, weights.x);
	weights.y = -weights.y;

	indices.x = sectionStartOffset + halfSectionOffset;
	indices.y = sectionStartOffset + halfSectionOffset + halfSectionWidth;

	if (passIndex == 0)
	{
		indices = reversebits(indices) >> (32 - BUTTERFLY_COUNT) & (LENGTH - 1);
	}
}

groupshared float3 pingPongArray[4][LENGTH];
void ButterflyPass(int passIndex, uint x, uint t0, uint t1, out float3 resultR, out float3 resultI)
{
	uint2 Indices;
	float2 Weights;
	GetButterflyValues(passIndex, x, Indices, Weights);

	float3 inputR1 = pingPongArray[t0][Indices.x];
	float3 inputI1 = pingPongArray[t1][Indices.x];

	float3 inputR2 = pingPongArray[t0][Indices.y];
	float3 inputI2 = pingPongArray[t1][Indices.y];

	resultR = (inputR1 + Weights.x * inputR2 + Weights.y * inputI2);
	resultI = (inputI1 - Weights.y * inputR2 + Weights.x * inputI2);
}

void ButterflyPassFinalNoI(int passIndex, int x, int t0, int t1, out float3 resultR)
{
	uint2 Indices;
	float2 Weights;
	GetButterflyValues(passIndex, x, Indices, Weights);

	float3 inputR1 = pingPongArray[t0][Indices.x];

	float3 inputR2 = pingPongArray[t0][Indices.y];
	float3 inputI2 = pingPongArray[t1][Indices.y];

	resultR = (inputR1 + Weights.x * inputR2 + Weights.y * inputI2);
}


[numthreads(LENGTH, 1, 1)]
void main(uint3 position : SV_DispatchThreadID)
{
	uint2 texturePos;
	if (ROWPASS)
		texturePos = uint2(position.xy);
	else
		texturePos = uint2(position.yx);

	// Load entire row or column into scratch array
	pingPongArray[0][position.x].xyz = FFTSourceR[texturePos];
	pingPongArray[1][position.x].xyz = FFTSourceI[texturePos];

	uint4 textureIndices = uint4(0, 1, 2, 3);


	for (int i = 0; i < BUTTERFLY_COUNT - 1; i++)
	{
		GroupMemoryBarrierWithGroupSync();
		ButterflyPass(i, position.x, textureIndices.x, textureIndices.y, pingPongArray[textureIndices.z][position.x].xyz, pingPongArray[textureIndices.w][position.x].xyz);
		textureIndices.xyzw = textureIndices.zwxy;
	}

	// Final butterfly will write directly to the target texture
	GroupMemoryBarrierWithGroupSync();

	// The final pass writes to the output UAV texture
	if (COLPASS)
		// last pass of the inverse transform. The imaginary value is no longer needed
		ButterflyPassFinalNoI(BUTTERFLY_COUNT - 1, position.x, textureIndices.x, textureIndices.y, FFTTargetR[texturePos]);
	else
		ButterflyPass(BUTTERFLY_COUNT - 1, position.x, textureIndices.x, textureIndices.y, FFTTargetR[texturePos], FFTTargetI[texturePos]);

}
