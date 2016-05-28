cbuffer fftBufer
{
	uint col_row_pass; // 0 - row, 1 - col
	float3 pad;
};

Texture2D<float4> butterfly : register(t1);

Texture2DArray<float4> FFTSource : register(t0);
RWTexture2DArray<float4> FFTTarget : register(u0);

// performs two FFTs on two inputs packed in a single texture
// returns two results packed in a single vec4
float4 fft2(int layer, float2 i, float2 w, uint3 DTid) {
	float4 input1 = FFTSource[uint3(DTid.y, i.x, layer)];
	float4 input2 = FFTSource[uint3(DTid.y, i.y, layer)];
	float res1x = w.x * input2.x - w.y * input2.y;
	float res1y = w.y * input2.x + w.x * input2.y;
	float res2x = w.x * input2.z - w.y * input2.w;
	float res2y = w.y * input2.z + w.x * input2.w;
	return input1 + float4(res1x, res1y, res2x, res2y);
}

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	float4 data = butterfly[uint2(DTid.y, col_row_pass)];
	float2 i = data.xy;
	float2 w = data.zw;

	FFTTarget[DTid] = fft2(DTid.z, i, w, DTid);
}