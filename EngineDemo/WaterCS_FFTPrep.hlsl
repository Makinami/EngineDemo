cbuffer TimeBufferCS
{
	float4 time;
};

struct FFTInitial
{
	float2 h_0;
	float2 h_0conj;
	float dispersion;
};

StructuredBuffer<FFTInitial> gFFTInput : register(t0);
RWTexture2D<float3> FFTPrepReal;
RWTexture2D<float3> FFTPrepImag;

//#define dx (255.0f/255)

static const float PI = 3.14159265f;

[numthreads(256, 1, 1)]
void main( int3 DTid : SV_DispatchThreadID )
{
	uint index = DTid.x * 256 + DTid.y;
	float cos_, sin_;
	sincos(gFFTInput[index].dispersion*time.x, sin_, cos_);
	float2 h_0 = gFFTInput[index].h_0;
	float2 h_0conj = gFFTInput[index].h_0conj;

	float2 hTilde = float2( cos_*(h_0.x+h_0conj.x)+sin_*(h_0conj.y-h_0.y), 
							cos_*(h_0.y+h_0conj.y) + sin_*(h_0.x-h_0conj.x) );
	
	float2 k = 2.0f*PI*(DTid.xy - 256.0f / 2.0f) / 256.0f;
	float len = length(k);

	float2 ddx = float2(0.0f, 0.0f);
	float2 ddz = float2(0.0f, 0.0f);
	if (len >= 0.000001f)
	{
		ddx = float2(hTilde.y*k.x / len, -hTilde.x*k.x / len);
		ddz = float2(hTilde.y*k.y / len, -hTilde.x*k.y / len);
	}

	FFTPrepReal[DTid.xy] = float3(ddx.x, hTilde.x, ddz.x);
	FFTPrepImag[DTid.xy] = float3(ddx.y, hTilde.y, ddz.y);
}