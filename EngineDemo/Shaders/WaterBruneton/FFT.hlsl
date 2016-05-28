#define FFT_SIZE 256
#define BUTTERFLY_COUNT 8

#define ROWPASS col_row_pass.x == 0
#define COLPASS col_row_pass.x == 1

cbuffer fftBufer
{
	uint col_row_pass; // 0 - row, 1 - col
	float3 pad;
};

Texture2DArray<float4> FFTSource : register(t0);
RWTexture2DArray<float4> FFTTarget : register(u0);

static const float PI = 3.14159265f;

void GetButterflyValues(uint passIndex, uint x, out uint2 indices, out float2 weights)
{
	int sectionWidth = 2 << passIndex;
	int halfSectionWidth = (uint)sectionWidth / 2;

	int sectionStartOffset = x & ~(sectionWidth - 1);
	int halfSectionOffset = x & (halfSectionWidth - 1);
	int sectionOffset = x & (sectionWidth - 1);

	sincos(2.0 * PI * sectionOffset / (float)sectionWidth, weights.y, weights.x);
	weights.y = -weights.y;

	indices.x = sectionStartOffset + halfSectionOffset;
	indices.y = sectionStartOffset + halfSectionOffset + halfSectionWidth;

	if (passIndex == 0)
	{
		indices = reversebits(indices) >> (32 - BUTTERFLY_COUNT) & (FFT_SIZE - 1);
	}
}

groupshared float4 pingPongArray[2][FFT_SIZE];

void ButterflyPass(int passIndex, uint x, uint t0, out float4 result)
{
	uint2 Indices;
	float2 Weights;
	GetButterflyValues(passIndex, x, Indices, Weights);

	float4 input1 = pingPongArray[t0][Indices.x];
	float4 input2 = pingPongArray[t0][Indices.y];

	float2 resultR = (input1.xz + Weights.x * input2.xz + Weights.y * input2.yw);
	float2 resultI = (input1.yw - Weights.y * input2.xz + Weights.x * input2.yw);

	result = float4(resultR.x, resultI.x, resultR.y, resultI.y);
}

[numthreads(FFT_SIZE, 1, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	uint2 texturePos;
	if (ROWPASS)
		texturePos = uint2(DTid.xy);
	else
		texturePos = uint2(DTid.yx);
	
	// Load etire row/col into scratch array
	pingPongArray[0][DTid.x] = FFTSource[uint3(texturePos, DTid.z)];

	uint2 textureIndices = uint2(0, 1);

	for (int i = 0; i < BUTTERFLY_COUNT; ++i)
	{
		GroupMemoryBarrierWithGroupSync();
		ButterflyPass(i, DTid.x, textureIndices.x, pingPongArray[textureIndices.y][DTid.x]);
		textureIndices.xy = textureIndices.yx;
	}

	FFTTarget[uint3(texturePos, DTid.z)] = pingPongArray[textureIndices.x][DTid.x];
}